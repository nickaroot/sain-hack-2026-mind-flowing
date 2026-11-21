// Голос экспертов: озвучивание реплик, очередь фрагментов и форма рта аватара под звучащий звук.
import { useCallback, useEffect, useRef, useState } from "react"
import { API_URL, speak } from "./api"
import { voiceMessage } from "./hooks"
import { mouthTrack, shapeAt } from "./visemes"
import type { MouthShape, MouthTrack } from "./visemes"

const params = new URLSearchParams(location.search)
/**
 * Длинный ответ озвучивается по предложениям: первое звучит через секунду-две, остальные готовятся, пока оно идёт.
 * Только для локального сервера голоса: у прокси организаторов 10 запросов синтеза в минуту, там ответ идёт одним запросом.
 */
const CHUNKED = import.meta.env.VITE_VOICE_CHUNKS === "1" || params.has("chunks")
/** Для проверки: ?blockaudio ведёт себя как браузер, который не даёт звук до первого касания. */
const SIMULATE_BLOCK = params.has("blockaudio")
// Короткая тишина: ею по первому касанию «будится» проигрыватель, чтобы дальше он мог включаться сам.
const SILENCE = "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQAAAAA="
const FRAME = 0.04 // шаг огибающей громкости, секунды
const MOUTH_LEAD = 0.06 // на сколько рот опережает звук, секунды

// Локальный стенд сам сообщает, можно ли дробить ответ: голос прода идёт одним запросом, локальные движки — по предложениям.
let chunked = CHUNKED
if (import.meta.env.VITE_LOCAL_STAND === "1") {
  const refresh = () =>
    void fetch(`${API_URL}/local/voice`)
      .then((response) => response.json())
      .then((settings) => {
        chunked = !!settings.chunks
      })
      .catch(() => undefined)
  refresh()
  window.addEventListener("focus", refresh)
}


/** Делит текст на фрагменты для синтеза по границам предложений; слова не теряются и не рвутся. */
export function splitForVoice(text: string): string[] {
  if (!chunked) return [text]
  const sentences = text.split(/(?<=[.!?…])\s+/).filter((part) => part.trim())
  // Длинное первое предложение делится по запятой: голос начинает раньше, а продолжение готовится на ходу.
  const head = sentences[0] ?? ""
  const comma = head.indexOf(", ", 24)
  if (head.length > 70 && comma > 0 && comma < head.length - 20) sentences.splice(0, 1, head.slice(0, comma + 1), head.slice(comma + 2))
  const chunks: string[] = []
  for (const sentence of sentences) {
    const last = chunks[chunks.length - 1]
    // Первый фрагмент короткий — чтобы голос начался быстрее; дальше склеиваем до удобной длины.
    const limit = chunks.length === 1 ? 24 : 110
    if (last !== undefined && last.length < limit) chunks[chunks.length - 1] = `${last} ${sentence}`
    else chunks.push(sentence)
  }
  return chunks.length ? chunks : [text]
}

/** Запись голоса и её огибающая: по громкости двигается рот аватара, по звучащим участкам — текст. */
type Clip = { url: string | null; level: Float32Array | null; voiced: Float32Array | null }

async function analyse(blob: Blob): Promise<Pick<Clip, "level" | "voiced">> {
  try {
    const Offline: typeof OfflineAudioContext =
      window.OfflineAudioContext ?? (window as unknown as { webkitOfflineAudioContext: typeof OfflineAudioContext }).webkitOfflineAudioContext
    const buffer = await new Offline(1, 1, 22050).decodeAudioData(await blob.arrayBuffer())
    const samples = buffer.getChannelData(0)
    const size = Math.max(1, Math.round(buffer.sampleRate * FRAME))
    const level = new Float32Array(Math.ceil(samples.length / size))
    let loudest = 0
    for (let frame = 0; frame < level.length; frame++) {
      let peak = 0
      for (let index = frame * size, end = Math.min(samples.length, index + size); index < end; index++) peak = Math.max(peak, Math.abs(samples[index]))
      level[frame] = peak
      loudest = Math.max(loudest, peak)
    }
    if (loudest === 0) return { level: null, voiced: null }
    // Текст идёт за речью: звучащий отрезок продвигает его на полный шаг, пауза — почти не продвигает.
    const voiced = new Float32Array(level.length)
    let sum = 0
    for (let frame = 0; frame < level.length; frame++) {
      level[frame] /= loudest
      sum += level[frame] > 0.08 ? 1 : 0.1
      voiced[frame] = sum
    }
    return { level, voiced }
  } catch {
    return { level: null, voiced: null }
  }
}

type Narration = {
  key: string
  /** Дорожки звуков по фрагментам: по ним рот аватара принимает форму произносимой буквы. */
  tracks: MouthTrack[]
  index: number
  done: boolean
  clip: Clip | null
  fake: { from: number; ms: number } | null
}

export function useSpeech() {
  const [loading, setLoading] = useState<string | null>(null)
  const [playing, setPlaying] = useState<string | null>(null)
  const [error, setError] = useState<{ key: string; message: string } | null>(null)
  const [unlocked, setUnlocked] = useState(false)
  // Браузер не дал начать звук без касания: ждём первого нажатия и не показываем текст немым.
  const [blocked, setBlocked] = useState(false)
  // Голос и текст эксперта неразрывны: реплика появляется вместе со звуком. Значок в шапке голос выключает.
  // При открытии страницы ничего не звучит и не синтезируется, пока человек не нажмёт «Послушать голос».
  const [auto, setAutoState] = useState(true)
  // Голос уже звучал в этом открытии страницы — дальше эксперт в галерее отвечает сам при выборе.
  const [started, setStarted] = useState(false)

  // Один проигрыватель на всё приложение. Звук идёт из него напрямую, без обработки: так он не может оказаться немым.
  const player = useRef<HTMLAudioElement | null>(null)
  const cache = useRef(new Map<string, Promise<Clip>>())
  const narration = useRef<Narration | null>(null)
  const token = useRef(0)
  const release = useRef<(() => void) | null>(null)
  const touched = useRef(false)

  const prepare = useCallback(() => {
    if (!player.current) {
      const element = new Audio()
      element.preload = "auto"
      player.current = element
    }
    return player.current
  }, [])

  /** Первое касание: с него браузер разрешает звук, дальше речь может начинаться сама. */
  const unlock = useCallback(() => {
    touched.current = true
    const element = prepare()
    if (!element.src) {
      element.src = SILENCE
      void element.play().catch(() => undefined)
    }
    setUnlocked(true)
    setBlocked(false)
  }, [prepare])

  useEffect(() => {
    const wake = () => unlock()
    document.addEventListener("pointerdown", wake, { once: true, capture: true })
    document.addEventListener("keydown", wake, { once: true, capture: true })
    return () => {
      document.removeEventListener("pointerdown", wake, { capture: true })
      document.removeEventListener("keydown", wake, { capture: true })
    }
  }, [unlock])

  const load = useCallback((text: string, expertId: string) => {
    const id = `${expertId}\n${text}`
    let ready = cache.current.get(id)
    if (!ready) {
      ready = speak(text, expertId).then(async (blob): Promise<Clip> => (blob ? { url: URL.createObjectURL(blob), ...(await analyse(blob)) } : { url: null, level: null, voiced: null }))
      cache.current.set(id, ready)
      ready.catch(() => cache.current.delete(id))
    }
    return ready
  }, [])

  /**
   * Ждёт, пока будет готово начало озвучки, но не дольше отведённого: текст и голос должны появиться вместе.
   * Отвечает, успел ли голос. Не успел — текст показывают сразу, а голос подключится, когда будет готов.
   */
  const stop = useCallback(() => {
    token.current += 1
    narration.current = null
    player.current?.pause()
    release.current?.()
    release.current = null
    if ("speechSynthesis" in window) speechSynthesis.cancel()
    setPlaying(null)
    setLoading(null)
  }, [])

  /** Громкость голоса в эту секунду (0…1) или null, если звука нет: по ней двигается рот аватара. */
  const level = useCallback((): number | null => {
    const clip = narration.current?.clip
    const element = player.current
    if (!clip?.level || !element || element.paused) return null
    return clip.level[Math.min(clip.level.length - 1, Math.floor(element.currentTime / FRAME))] ?? 0
  }, [])

  /**
   * Форма рта в эту секунду: какая буква реплики сейчас звучит и насколько громко. null — звука нет.
   * Положение в тексте считается по звучащим участкам записи, как и у проявления слов.
   */
  const mouth = useCallback((): MouthShape | null => {
    const current = narration.current
    const element = player.current
    if (!current || current.done || current.index < 0) return null
    const track = current.tracks[current.index]
    if (current.fake) return shapeAt(track, (performance.now() - current.fake.from) / current.fake.ms)
    if (!element || element.paused) return null
    const clip = current.clip
    if (!clip?.voiced || !clip.level) return element.duration > 0 ? shapeAt(track, element.currentTime / element.duration) : null
    // Рот опережает звук на пару кадров: так делают в мультипликации, иначе губы будто догоняют голос.
    const frame = Math.min(clip.voiced.length - 1, Math.floor((element.currentTime + MOUTH_LEAD) / FRAME))
    const loud = clip.level[frame] ?? 0
    const shape = shapeAt(track, clip.voiced[frame] / clip.voiced[clip.voiced.length - 1])
    // В тишине рот закрыт, какой бы ни была буква; тихий звук открывает его меньше громкого.
    if (loud < 0.06) return { open: 0, round: shape.round * 0.4 }
    return { open: shape.open * (0.55 + 0.45 * Math.min(1, loud / 0.6)), round: shape.round }
  }, [])

  const narrate = useCallback(
    async (key: string, text: string, expertId: string) => {
      stop()
      const element = prepare()
      const mine = token.current
      const chunks = splitForVoice(text)
      const current: Narration = { key, tracks: chunks.map(mouthTrack), index: -1, done: false, clip: null, fake: null }
      narration.current = current
      setError(null)
      setLoading(key)
      const ended = () =>
        new Promise<void>((resolve) => {
          release.current = resolve
        })
      try {
        let next = load(chunks[0], expertId)
        for (let index = 0; index < chunks.length; index++) {
          const clip = await next
          if (mine !== token.current) return
          if (index + 1 < chunks.length) next = load(chunks[index + 1], expertId)
          if (clip.url) {
            if (SIMULATE_BLOCK && !touched.current) throw new DOMException("звук до касания запрещён", "NotAllowedError")
            element.muted = false
            element.volume = 1
            element.src = clip.url
            const finished = ended()
            element.onended = () => release.current?.()
            element.onerror = () => release.current?.()
            await element.play()
            // Звук пошёл — значит, браузер его разрешил (сразу или после касания).
            setUnlocked(true)
            setBlocked(false)
            setStarted(true)
            current.fake = null
            current.clip = clip
            current.index = index
            setLoading(null)
            setPlaying(key)
            await finished
          } else {
            // Заглушки без звукового файла: говорит синтезатор браузера, время считаем по длине текста.
            const ms = Math.min(30_000, 900 + chunks[index].length * 62)
            setStarted(true)
            current.fake = { from: performance.now(), ms }
            current.clip = null
            current.index = index
            setLoading(null)
            setPlaying(key)
            if ("speechSynthesis" in window) {
              const utterance = new SpeechSynthesisUtterance(chunks[index])
              utterance.lang = "ru-RU"
              speechSynthesis.speak(utterance)
            }
            const finished = ended()
            const timer = setTimeout(() => release.current?.(), ms)
            await finished
            clearTimeout(timer)
          }
          if (mine !== token.current) return
        }
        current.done = true
        setPlaying(null)
      } catch (reason) {
        if (mine !== token.current) return
        narration.current = null
        setLoading(null)
        setPlaying(null)
        // Браузер не дал начать звук без касания — это не ошибка сервиса, просто ждём нажатия.
        if (reason instanceof DOMException && reason.name === "NotAllowedError") setBlocked(true)
        else setError({ key, message: voiceMessage(reason) })
      }
    },
    [load, prepare, stop],
  )

  /** Нажатие на «озвучить»: включает чтение, а если эта реплика уже звучит — останавливает. */
  const toggle = useCallback(
    (key: string, text: string, expertId: string) => {
      if (playing === key || loading === key) stop()
      else void narrate(key, text, expertId)
    },
    [loading, narrate, playing, stop],
  )

  const setAuto = useCallback(
    (value: boolean) => {
      setAutoState(value)
      if (!value) stop()
    },
    [stop],
  )

  useEffect(() => stop, [stop])
  return { loading, playing, error, auto, started, unlocked, blocked, setAuto, toggle, narrate, stop, level, mouth }
}

export type Speech = ReturnType<typeof useSpeech>
