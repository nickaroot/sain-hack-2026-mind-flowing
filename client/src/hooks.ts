import { useCallback, useEffect, useRef, useState } from "react"
import { ApiError, MOCK, transcribe } from "./api"

export type Theme = "light" | "dark"

export function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(() => (document.documentElement.dataset.theme === "dark" ? "dark" : "light"))
  const toggle = useCallback(() => {
    setTheme((current) => {
      const next = current === "dark" ? "light" : "dark"
      document.documentElement.dataset.theme = next
      try {
        localStorage.setItem("mf-theme", next)
      } catch {
        // хранилище недоступно — тема просто не запомнится
      }
      return next
    })
  }, [])
  return [theme, toggle]
}

export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => matchMedia(query).matches)
  useEffect(() => {
    const list = matchMedia(query)
    const update = () => setMatches(list.matches)
    update()
    list.addEventListener("change", update)
    // Не все окружения сообщают о смене ширины через медиазапрос, поэтому слушаем и само окно.
    window.addEventListener("resize", update)
    return () => {
      list.removeEventListener("change", update)
      window.removeEventListener("resize", update)
    }
  }, [query])
  return matches
}

/** Сколько секунд прошло с момента `since`; пока `active`, обновляется раз в секунду. */
export function useElapsed(since: number, active: boolean): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!active) return
    setNow(Date.now())
    const timer = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(timer)
  }, [active, since])
  return Math.max(0, Math.floor((now - since) / 1000))
}

export const voiceMessage = (error: unknown): string => {
  if (error instanceof ApiError) {
    if (error.kind === "quota") return "Голос на сегодня закончился — читать и писать можно как обычно."
    if (error.kind === "busy") return "Слишком много запросов к голосу. Попробуйте через несколько секунд."
    if (error.kind === "paused") return "Сервер занят. Попробуйте через минуту."
    if (error.kind === "network") return "Нет связи с сервером."
    if (error.status === 413) return "Запись слишком длинная."
  }
  return "Голос сейчас недоступен. Попробуйте ещё раз."
}

export type RecorderState = "idle" | "recording" | "transcribing"

const pickMime = () =>
  ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"].find(
    (type) => typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(type),
  )

// Распознавание идёт по ходу диктовки: на паузе в речи записанный кусок сразу уходит на сервер, поэтому к концу
// диктовки ждать остаётся только последний кусок, а если человек замолчал перед нажатием — ничего.
// Каждый кусок — отдельный запрос, а у сервера суточный лимит распознаваний и общий лимит запросов в минуту,
// поэтому режем только по настоящим паузам, не слишком мелко и не больше MAX_PIECES раз за диктовку.
/** Уровень микрофона (0…1), ниже которого считаем, что человек молчит. */
const VOICE_LEVEL = 0.12
/** Пауза, по которой кусок уходит на распознавание, мс. */
const PAUSE_MS = 700
/** Куски короче не отрезаем, мс. */
const MIN_PIECE_MS = 2500
/** Кусок, в котором речи меньше, не отправляем вовсе, мс. */
const MIN_VOICE_MS = 350
const MAX_PIECES = 8
const MAX_PIECE_BYTES = 5 * 1024 * 1024

type Part = { text: string; error?: unknown }
type Session = {
  stream: MediaStream | null
  context: AudioContext | null
  frame: number
  mime: string | undefined
  recorder: MediaRecorder | null
  chunks: Blob[]
  pieceStarted: number
  /** Сколько в текущем куске речи и сколько длится текущая тишина, мс. */
  voiced: number
  quiet: number
  /** Распознанные и распознающиеся куски по порядку. */
  parts: Promise<Part>[]
  cuts: number
  sealing: Promise<void> | null
  closing: boolean
}

const recognise = (blob: Blob): Promise<Part> =>
  blob.size > MAX_PIECE_BYTES
    ? Promise.resolve({ text: "", error: new ApiError("invalid", "Запись слишком длинная", 413) })
    : transcribe(blob).then(
        (text) => ({ text: text.trim() }),
        (error) => ({ text: "", error }),
      )

/**
 * Запись вопроса голосом: микрофон → POST /api/transcribe по кускам → текст.
 * `finish` кладёт распознанный текст в поле ввода, `send` сразу отправляет его в разговор.
 */
export function useRecorder(onText: (text: string) => void, onSend: (text: string) => void) {
  const supported = MOCK || (typeof MediaRecorder !== "undefined" && !!navigator.mediaDevices?.getUserMedia)
  const [state, setState] = useState<RecorderState>("idle")
  const [level, setLevel] = useState(0)
  const [startedAt, setStartedAt] = useState(0)
  const [error, setError] = useState<string | null>(null)
  /** Чем закончили диктовку: по этому кнопка показывает, что именно сейчас происходит. */
  const [ending, setEnding] = useState<"insert" | "send" | null>(null)
  const session = useRef<Session | null>(null)
  const handlers = useRef({ onText, onSend })
  handlers.current = { onText, onSend }

  const release = useCallback(() => {
    const current = session.current
    if (!current) return
    cancelAnimationFrame(current.frame)
    current.stream?.getTracks().forEach((track) => track.stop())
    void current.context?.close().catch(() => undefined)
    session.current = null
    setLevel(0)
  }, [])

  /** Начать новый кусок записи на том же микрофоне. */
  const open = (current: Session) => {
    if (!current.stream) return
    const recorder = new MediaRecorder(current.stream, current.mime ? { mimeType: current.mime } : undefined)
    const chunks: Blob[] = []
    recorder.ondataavailable = (event) => event.data.size > 0 && chunks.push(event.data)
    recorder.start(250)
    current.recorder = recorder
    current.chunks = chunks
    current.pieceStarted = performance.now()
    current.voiced = 0
    current.quiet = 0
  }

  /** Закрыть текущий кусок; если в нём была речь и он нужен — сразу отправить на распознавание. */
  const seal = (current: Session, keep: boolean): Promise<void> => {
    const { recorder, chunks, voiced } = current
    current.recorder = null
    if (!recorder) return Promise.resolve()
    return new Promise((resolve) => {
      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" })
        if (keep && voiced >= MIN_VOICE_MS && blob.size > 0) current.parts.push(recognise(blob))
        resolve()
      }
      recorder.stop()
    })
  }

  const start = useCallback(async () => {
    setError(null)
    let stream: MediaStream | null = null
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    } catch {
      if (!MOCK) {
        setError("Нет доступа к микрофону. Разрешите его в браузере или напишите вопрос текстом.")
        return
      }
    }
    const current: Session = {
      stream,
      context: null,
      frame: 0,
      mime: stream ? pickMime() : undefined,
      recorder: null,
      chunks: [],
      pieceStarted: 0,
      voiced: 0,
      quiet: 0,
      parts: [],
      cuts: 0,
      sealing: null,
      closing: false,
    }
    session.current = current
    if (stream) {
      open(current)
      const context = new AudioContext()
      current.context = context
      const analyser = context.createAnalyser()
      analyser.fftSize = 256
      context.createMediaStreamSource(stream).connect(analyser)
      const data = new Uint8Array(analyser.fftSize)
      let last = performance.now()
      const tick = () => {
        analyser.getByteTimeDomainData(data)
        let peak = 0
        for (const value of data) peak = Math.max(peak, Math.abs(value - 128))
        const heard = Math.min(1, peak / 64)
        setLevel(heard)
        const now = performance.now()
        const dt = Math.min(250, now - last)
        last = now
        if (heard >= VOICE_LEVEL) {
          current.voiced += dt
          current.quiet = 0
        } else {
          current.quiet += dt
        }
        const ripe = current.quiet >= PAUSE_MS && current.voiced >= MIN_VOICE_MS && now - current.pieceStarted >= MIN_PIECE_MS
        if (ripe && current.recorder && !current.sealing && !current.closing && current.cuts < MAX_PIECES) {
          current.cuts++
          current.sealing = seal(current, true).then(() => {
            current.sealing = null
            if (session.current === current && !current.closing) open(current)
          })
        }
        current.frame = requestAnimationFrame(tick)
      }
      tick()
    } else {
      // Заглушка без микрофона: уровень «дышит», чтобы экран записи можно было проверить.
      const tick = () => {
        setLevel(0.35 + 0.35 * Math.sin(performance.now() / 180))
        current.frame = requestAnimationFrame(tick)
      }
      tick()
    }
    setStartedAt(Date.now())
    setState("recording")
  }, [])

  const cancel = useCallback(() => {
    const current = session.current
    if (current) {
      current.closing = true
      void seal(current, false)
    }
    release()
    setState("idle")
  }, [release])

  const end = useCallback(
    async (mode: "insert" | "send") => {
      const current = session.current
      if (!current || current.closing) return
      current.closing = true
      setEnding(mode)
      setState("transcribing")
      try {
        await current.sealing
        await seal(current, true)
        if (!current.stream) current.parts.push(recognise(new Blob([], { type: "audio/webm" })))
        release()
        const parts = await Promise.all(current.parts)
        const failed = parts.find((part) => part.error)
        const text = parts
          .map((part) => part.text)
          .filter(Boolean)
          .join(" ")
        if (!text) {
          setError(failed ? voiceMessage(failed.error) : "Не удалось разобрать речь. Попробуйте ещё раз или напишите вопрос.")
        } else if (failed) {
          // Отправлять вопрос с пропуском нельзя: текст остаётся в поле, человек его проверит.
          handlers.current.onText(text)
          setError("Часть записи не распозналась — проверьте текст перед отправкой.")
        } else if (mode === "send") {
          handlers.current.onSend(text)
        } else {
          handlers.current.onText(text)
        }
      } finally {
        release()
        setState("idle")
        setEnding(null)
      }
    },
    [release],
  )
  const finish = useCallback(() => end("insert"), [end])
  const send = useCallback(() => end("send"), [end])

  // Запись длиннее двух минут не нужна вопросу.
  useEffect(() => {
    if (state !== "recording") return
    const timer = setTimeout(() => void finish(), 120_000)
    return () => clearTimeout(timer)
  }, [state, finish])

  useEffect(() => release, [release])
  return { supported, state, level, startedAt, error, ending, start, cancel, finish, send, clearError: () => setError(null) }
}
