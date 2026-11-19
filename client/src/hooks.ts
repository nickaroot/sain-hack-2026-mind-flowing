import { useCallback, useEffect, useRef, useState } from "react"
import { ApiError, MOCK, speak, transcribe } from "./api"

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
    return () => list.removeEventListener("change", update)
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

const voiceMessage = (error: unknown): string => {
  if (error instanceof ApiError) {
    if (error.kind === "quota") return "Голос на сегодня закончился — читать и писать можно как обычно."
    if (error.kind === "busy") return "Слишком много запросов к голосу. Попробуйте через несколько секунд."
    if (error.kind === "paused") return "Сервер занят. Попробуйте через минуту."
    if (error.kind === "network") return "Нет связи с сервером."
    if (error.status === 413) return "Запись слишком длинная."
  }
  return "Голос сейчас недоступен. Попробуйте ещё раз."
}

/** Озвучивание по нажатию. Ключ — что именно звучит: одна реплика одного эксперта. */
export function useSpeech() {
  const [loading, setLoading] = useState<string | null>(null)
  const [playing, setPlaying] = useState<string | null>(null)
  const [error, setError] = useState<{ key: string; message: string } | null>(null)
  const audio = useRef<HTMLAudioElement | null>(null)
  const cache = useRef(new Map<string, string>())
  const token = useRef(0)
  const fake = useRef<ReturnType<typeof setTimeout> | null>(null)
  // Громкость звучащего голоса: по ней аватар шевелит ртом. Без неё звук играет как обычно.
  const context = useRef<AudioContext | null>(null)
  const meter = useRef<{ node: AnalyserNode; data: Uint8Array<ArrayBuffer> } | null>(null)

  const level = useCallback((): number | null => {
    const current = meter.current
    if (!current || !audio.current || audio.current.paused) return null
    current.node.getByteTimeDomainData(current.data)
    let peak = 0
    for (const value of current.data) peak = Math.max(peak, Math.abs(value - 128))
    return Math.min(1, peak / 48)
  }, [])

  const stop = useCallback(() => {
    token.current += 1
    audio.current?.pause()
    audio.current = null
    meter.current = null
    if (fake.current) clearTimeout(fake.current)
    if ("speechSynthesis" in window) speechSynthesis.cancel()
    setPlaying(null)
    setLoading(null)
  }, [])

  const toggle = useCallback(
    async (key: string, text: string, expertId: string) => {
      const wasActive = playing === key || loading === key
      stop()
      if (wasActive) return
      const mine = token.current
      setError(null)
      setLoading(key)
      // Звуковой контекст будим сразу по нажатию: позже браузер может этого не разрешить.
      try {
        context.current ??= new AudioContext()
        void context.current.resume()
      } catch {
        context.current = null
      }
      try {
        let url = cache.current.get(key) ?? null
        if (!url) {
          const blob = await speak(text, expertId)
          if (mine !== token.current) return
          if (blob) {
            url = URL.createObjectURL(blob)
            cache.current.set(key, url)
          }
        }
        setLoading(null)
        setPlaying(key)
        const done = () => mine === token.current && setPlaying(null)
        if (url) {
          const player = new Audio(url)
          audio.current = player
          const ctx = context.current
          if (ctx && ctx.state === "running") {
            try {
              const node = ctx.createAnalyser()
              node.fftSize = 512
              ctx.createMediaElementSource(player).connect(node)
              node.connect(ctx.destination)
              meter.current = { node, data: new Uint8Array(new ArrayBuffer(node.fftSize)) }
            } catch {
              meter.current = null
            }
          }
          player.onended = done
          player.onerror = done
          await player.play()
        } else if ("speechSynthesis" in window) {
          // Заглушка: говорит синтезатор браузера, чтобы состояние «звучит» было видно без стенда.
          const utterance = new SpeechSynthesisUtterance(text)
          utterance.lang = "ru-RU"
          utterance.onend = done
          utterance.onerror = done
          speechSynthesis.speak(utterance)
          fake.current = setTimeout(done, Math.min(30_000, 1500 + text.length * 70))
        } else {
          fake.current = setTimeout(done, Math.min(12_000, 1500 + text.length * 40))
        }
      } catch (reason) {
        if (mine !== token.current) return
        setLoading(null)
        setPlaying(null)
        setError({ key, message: voiceMessage(reason) })
      }
    },
    [loading, playing, stop],
  )

  useEffect(() => stop, [stop])
  return { loading, playing, error, toggle, stop, level }
}

export type RecorderState = "idle" | "recording" | "transcribing"

const pickMime = () =>
  ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"].find(
    (type) => typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(type),
  )

/** Запись вопроса голосом: микрофон → POST /api/transcribe → текст в поле ввода. */
export function useRecorder(onText: (text: string) => void) {
  const supported = MOCK || (typeof MediaRecorder !== "undefined" && !!navigator.mediaDevices?.getUserMedia)
  const [state, setState] = useState<RecorderState>("idle")
  const [level, setLevel] = useState(0)
  const [startedAt, setStartedAt] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const session = useRef<{ recorder: MediaRecorder | null; stream: MediaStream | null; context: AudioContext | null; frame: number; chunks: Blob[] } | null>(null)

  const release = useCallback(() => {
    const current = session.current
    if (!current) return
    cancelAnimationFrame(current.frame)
    current.stream?.getTracks().forEach((track) => track.stop())
    void current.context?.close().catch(() => undefined)
    session.current = null
    setLevel(0)
  }, [])

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
    const current = { recorder: null as MediaRecorder | null, stream, context: null as AudioContext | null, frame: 0, chunks: [] as Blob[] }
    session.current = current
    if (stream) {
      const mimeType = pickMime()
      current.recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
      current.recorder.ondataavailable = (event) => event.data.size > 0 && current.chunks.push(event.data)
      current.recorder.start(250)
      const context = new AudioContext()
      current.context = context
      const analyser = context.createAnalyser()
      analyser.fftSize = 256
      context.createMediaStreamSource(stream).connect(analyser)
      const data = new Uint8Array(analyser.fftSize)
      const tick = () => {
        analyser.getByteTimeDomainData(data)
        let peak = 0
        for (const value of data) peak = Math.max(peak, Math.abs(value - 128))
        setLevel(Math.min(1, peak / 64))
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
    session.current?.recorder?.stop()
    release()
    setState("idle")
  }, [release])

  const finish = useCallback(async () => {
    const current = session.current
    if (!current) return
    setState("transcribing")
    try {
      let blob = new Blob([], { type: "audio/webm" })
      if (current.recorder) {
        const recorder = current.recorder
        await new Promise<void>((resolve) => {
          recorder.onstop = () => resolve()
          recorder.stop()
        })
        blob = new Blob(current.chunks, { type: recorder.mimeType || "audio/webm" })
      }
      release()
      if (!MOCK && blob.size === 0) throw new ApiError("invalid", "Пустая запись", 422)
      if (blob.size > 5 * 1024 * 1024) throw new ApiError("invalid", "Запись слишком длинная", 413)
      const text = (await transcribe(blob)).trim()
      if (text) onText(text)
      else setError("Не удалось разобрать речь. Попробуйте ещё раз или напишите вопрос.")
    } catch (reason) {
      release()
      setError(reason instanceof ApiError && reason.status === 422 ? "Запись получилась пустой. Попробуйте ещё раз." : voiceMessage(reason))
    } finally {
      setState("idle")
    }
  }, [onText, release])

  // Запись длиннее двух минут не нужна вопросу и приближается к пределу сервера.
  useEffect(() => {
    if (state !== "recording") return
    const timer = setTimeout(() => void finish(), 120_000)
    return () => clearTimeout(timer)
  }, [state, finish])

  useEffect(() => release, [release])
  return { supported, state, level, startedAt, error, start, cancel, finish, clearError: () => setError(null) }
}
