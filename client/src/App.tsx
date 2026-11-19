import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { ChevronDown, Moon, Plus, Sun } from "lucide-react"
import { ApiError, MOCK, chat, getExperts } from "./api"
import type { Citation } from "./api"
import { historyFor } from "./chat"
import type { Turn } from "./chat"
import { accentStyle, persona } from "./experts"
import type { Persona } from "./experts"
import { useMediaQuery, useRecorder, useSpeech, useTheme } from "./hooks"
import { Avatar } from "./components/Avatar"
import { Composer } from "./components/Composer"
import { ExpertAbout, ExpertPicker } from "./components/ExpertSheets"
import { Gallery } from "./components/Gallery"
import { SourceView } from "./components/Source"
import { Thread } from "./components/Thread"

// Темы примеров есть в базе статей (кофеин, экран вечером, дневной сон). Последний — у границы:
// на него эксперт мягко отказывает, и это выглядит как обычная реплика.
const SUGGESTIONS = [
  "Мешает ли кофе после обеда уснуть?",
  "Вреден ли телефон перед сном?",
  "Помогает ли короткий дневной сон?",
  "Сколько мелатонина мне пить?",
]

const remembered = () => {
  try {
    return localStorage.getItem("mf-expert") ?? ""
  } catch {
    return ""
  }
}

export function App() {
  const [theme, toggleTheme] = useTheme()
  const wide = useMediaQuery("(min-width: 1024px)")
  const roomy = useMediaQuery("(min-width: 720px)")
  const [personas, setPersonas] = useState<Persona[] | null>(null)
  const [failed, setFailed] = useState(false)
  const [activeId, setActiveId] = useState(remembered)
  const [turns, setTurns] = useState<Turn[]>([])
  const [draft, setDraft] = useState("")
  const [source, setSource] = useState<Citation | null>(null)
  const [picker, setPicker] = useState(false)
  const [about, setAbout] = useState(false)
  const speech = useSpeech()
  const recorder = useRecorder(useCallback((text: string) => setDraft((current) => (current ? `${current} ${text}` : text)), []))

  const turnsRef = useRef(turns)
  turnsRef.current = turns
  const activeRef = useRef(activeId)
  activeRef.current = activeId

  // Эксперты приходят с сервера: новый появится в галерее без правок клиента.
  const load = useCallback(() => {
    setFailed(false)
    getExperts()
      .then((data) => {
        const list = data.experts.map(persona)
        setPersonas(list)
        setActiveId((current) => (list.some((item) => item.id === current) ? current : (list[0]?.id ?? "")))
      })
      .catch(() => setFailed(true))
  }, [])
  useEffect(load, [load])

  const active = useMemo(() => personas?.find((item) => item.id === activeId) ?? personas?.[0] ?? null, [personas, activeId])
  const pending = turns.some((turn) => turn.status === "pending")
  const home = turns.length === 0

  const patch = (id: string, change: Partial<Turn>) => setTurns((list) => list.map((turn) => (turn.id === id ? { ...turn, ...change } : turn)))

  const run = useCallback(async (turn: Turn, before: Turn[], expertId: string) => {
    try {
      const response = await chat(expertId, turn.question, historyFor(before, expertId))
      patch(turn.id, { status: "done", response, error: undefined, fresh: true })
    } catch (reason) {
      patch(turn.id, { status: "error", error: reason instanceof ApiError ? reason : new ApiError("unknown", "") })
    }
  }, [])

  const ask = useCallback(
    (text: string) => {
      const question = text.trim()
      if (!question || !activeRef.current || turnsRef.current.some((turn) => turn.status === "pending")) return
      const turn: Turn = { id: crypto.randomUUID(), question, status: "pending", startedAt: Date.now(), attempts: 1 }
      const before = turnsRef.current
      setTurns([...before, turn])
      setDraft("")
      void run(turn, before, activeRef.current)
    },
    [run],
  )

  // Повтор шлёт тот же вопрос: сервер дождётся уже начатого расчёта и второй раз его не посчитает.
  const retry = useCallback(
    (id: string) => {
      const list = turnsRef.current
      const position = list.findIndex((turn) => turn.id === id)
      if (position < 0 || list.some((turn) => turn.status === "pending")) return
      const turn: Turn = { ...list[position], status: "pending", startedAt: Date.now(), attempts: list[position].attempts + 1, error: undefined, response: undefined }
      patch(id, turn)
      void run(turn, list.slice(0, position), activeRef.current)
    },
    [run],
  )

  const revealed = useCallback((id: string) => patch(id, { fresh: false }), [])

  const pick = useCallback(
    (id: string) => {
      speech.stop()
      setActiveId(id)
      try {
        localStorage.setItem("mf-expert", id)
      } catch {
        // выбор просто не запомнится
      }
    },
    [speech],
  )

  const reset = () => {
    speech.stop()
    setTurns([])
    setSource(null)
  }

  // Новый вопрос поднимается к верху экрана: ответ появится под ним и не убежит из-под глаз.
  const elements = useRef(new Map<string, HTMLElement>())
  const register = useCallback((id: string, element: HTMLElement | null) => {
    if (element) elements.current.set(id, element)
    else elements.current.delete(id)
  }, [])
  const lastId = turns[turns.length - 1]?.id
  useEffect(() => {
    if (lastId) elements.current.get(lastId)?.scrollIntoView({ block: "start", behavior: "smooth" })
  }, [lastId])

  if (!personas || !active) {
    return (
      <div className="app" data-screen="home">
        <div className="boot" role="status">
          {failed ? (
            <>
              <strong>Не удалось получить список экспертов</strong>
              <span>Проверьте связь с сервером и попробуйте ещё раз.</span>
              <button type="button" className="button-solid" onClick={load}>
                Повторить
              </button>
            </>
          ) : (
            <span className="shimmer">Знакомлюсь с экспертами…</span>
          )}
        </div>
      </div>
    )
  }

  return (
    <div className="app" data-screen={home ? "home" : "chat"} style={accentStyle(active.hue)}>
      <header className="top">
        <button type="button" className="brand" onClick={reset} aria-label="Mind-flowing — к выбору эксперта">
          <Moon size={22} aria-hidden="true" />
          <span>Mind-flowing</span>
        </button>
        {!home && (
          <button type="button" className="expert-pill" onClick={() => setPicker(true)} aria-haspopup="dialog">
            <Avatar persona={active} size={24} state="active" />
            <span className="expert-pill-full">{active.name}</span>
            <span className="expert-pill-short">{active.short}</span>
            <ChevronDown size={14} strokeWidth={2.4} aria-hidden="true" />
          </button>
        )}
        <span className="top-space" />
        {MOCK && <span className="mock-badge">заглушки</span>}
        {!home && (
          <button type="button" className="button-outline new-chat" onClick={reset} aria-label="Новый разговор">
            <Plus size={16} aria-hidden="true" />
            <span>Новый разговор</span>
          </button>
        )}
        <button type="button" className="icon-button" onClick={toggleTheme} aria-label={theme === "dark" ? "Светлая тема" : "Тёмная тема"}>
          {theme === "dark" ? <Sun size={20} /> : <Moon size={20} />}
        </button>
      </header>

      <div className="body">
        <main className="main">
          <div className="scroll">
            {home ? (
              <Gallery personas={personas} activeId={active.id} onActivate={pick} onAbout={() => setAbout(true)} speech={speech} />
            ) : (
              <Thread
                turns={turns}
                personas={personas}
                active={active}
                openSource={source}
                onSwitch={pick}
                onRetry={retry}
                onSource={(citation) => setSource((current) => (current?.article_id === citation.article_id && current.section_id === citation.section_id ? null : citation))}
                onRevealed={revealed}
                speech={speech}
                registerTurn={register}
              />
            )}
          </div>
          <div className="dock">
            {home && (
              <div className="suggestions">
                {SUGGESTIONS.map((text) => (
                  <button key={text} type="button" className="chip" onClick={() => ask(text)}>
                    {text}
                  </button>
                ))}
              </div>
            )}
            <Composer
              value={draft}
              onChange={setDraft}
              onSubmit={() => ask(draft)}
              pending={pending}
              recorder={recorder}
              placeholder={
                home
                  ? roomy
                    ? `Ваш вопрос о сне — ответит ${active.short.toLowerCase()}`
                    : "Спросите о сне…"
                  : pending
                    ? "Следующий вопрос можно набирать…"
                    : "Уточните или спросите ещё…"
              }
            />
            <p className="disclaimer">Эксперты вымышленные и не ставят диагнозы.</p>
          </div>
        </main>
        <SourceView citation={source} wide={wide} onClose={() => setSource(null)} />
      </div>

      <ExpertPicker open={picker} personas={personas} activeId={active.id} onPick={pick} onClose={() => setPicker(false)} />
      <ExpertAbout open={about} persona={active} onClose={() => setAbout(false)} />
    </div>
  )
}
