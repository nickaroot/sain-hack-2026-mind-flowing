import { useEffect, useMemo, useRef } from "react"

type Props = {
  text: string
  /** Темп эксперта: больше — медленнее. */
  pace?: number
  /** Текст появился целиком. */
  onDone?: () => void
}

/** Фраза не длиннее строки субтитров: столько глаз берёт за одну-две фиксации. */
const PHRASE_CHARS = 38
/**
 * Скорость появления, букв в секунду. Текст не подстраивается под голос: читают в несколько раз быстрее, чем
 * слушают, поэтому он идёт вперёд. Быстрый читатель берёт 40–45 букв в секунду — текст должен опережать его
 * с запасом, иначе глаз упирается в край. Для сравнения: ChatGPT выводит 14–19 слов в секунду (около 110–130
 * букв), в записи Gemini — около 400 букв в секунду. Здесь между ними: читателя текст опережает впятеро, а на
 * коротком ответе волна успевает быть видна.
 */
const SPEED = 220
/** Короткая реплика появляется не быстрее чем за столько миллисекунд (но и не медленнее MIN_SPEED): иначе волну не разглядеть. */
const MIN_TOTAL = 1400
const MIN_SPEED = 120
/**
 * Сколько длится анимация слова, мс (она же — в styles.css). Непрозрачным слово становится за первую треть,
 * около 300 мс: у Gemini кусок набирает цвет за 250–300 мс, у ChatGPT слово — за 275 мс. Остальное время
 * оттенок эксперта, с которым слово появилось, уходит в обычный цвет текста — так же делает ChatGPT.
 */
const FADE = 900

/** Слова абзаца → фразы. Возвращает, с какого слова начинается каждая фраза. */
function phraseStarts(words: string[]): number[] {
  const size = (from: number, to: number) => words.slice(from, to).join(" ").length
  // 1. Режем по паузам речи: после знака препинания и перед тире или скобкой.
  const clauses: [from: number, to: number][] = []
  let from = 0
  words.forEach((word, index) => {
    const next = words[index + 1]
    if (next === undefined || /[,.;:!?…»)]$/.test(word) || /^[—–(«]/.test(next)) {
      clauses.push([from, index + 1])
      from = index + 1
    }
  })
  // 2. Короткие соседние куски склеиваем, пока фраза помещается в строку: «Может мешать, да.» — одна фраза.
  //    Через конец предложения склеиваем только совсем короткое.
  const merged: [number, number][] = []
  for (const clause of clauses) {
    const last = merged[merged.length - 1]
    const ends = last ? /[.!?…]$/.test(words[last[1] - 1]) : false
    const tiny = last ? Math.min(size(...last), size(...clause)) < 9 : false
    if (last && size(last[0], clause[1]) <= PHRASE_CHARS && (!ends || tiny)) last[1] = clause[1]
    else merged.push([...clause])
  }
  // 3. Длинное без знаков делим на равные по длине части по границам слов.
  const starts: number[] = []
  for (const [begin, end] of merged) {
    const parts = Math.ceil(size(begin, end) / PHRASE_CHARS)
    const share = size(begin, end) / parts
    starts.push(begin)
    let cut = 1
    for (let index = begin + 1; index < end && cut < parts; index++) {
      if (size(begin, index) >= share * cut && end - index >= 2) {
        starts.push(index)
        cut++
      }
    }
  }
  return starts
}

/**
 * Текст, который появляется фразами, как в Gemini: фраза встаёт на место целиком, бледной, и по ней слева
 * направо проходит волна — слова набирают цвет. Волна идёт с постоянной скоростью через весь текст, без
 * остановок между фразами. Глаз получает готовый кусок смысла, а не бегущий край из букв; уже показанный
 * текст не шевелится, к нему можно вернуться взглядом.
 * Место под весь текст занято сразу, поэтому строки не прыгают и то, что ниже, не сдвигается.
 * Проявление — анимация в CSS; цикл кадров только решает, какую фразу пора показать.
 */
export function TypedText({ text, pace = 1, onDone }: Props) {
  const root = useRef<HTMLDivElement>(null)
  const done = useRef(onDone)
  done.current = onDone

  // Абзацы → фразы → слова. У каждой фразы запомнено, сколько букв текста стоит перед ней, у слова — перед ним во фразе.
  const { paragraphs, phrases, total } = useMemo(() => {
    const phrases: { before: number; chars: number }[] = []
    let before = 0
    const paragraphs = text
      .split(/\n{2,}/)
      .map((block) => block.trim())
      .filter(Boolean)
      .map((block) => {
        const all = block.split(/\s+/)
        const starts = phraseStarts(all)
        return starts.map((start, index) => {
          let inside = 0
          const part = all.slice(start, starts[index + 1] ?? all.length).map((word) => {
            const entry = { word, offset: inside }
            inside += word.length + 1
            return entry
          })
          phrases.push({ before, chars: inside })
          before += inside
          return part
        })
      })
    return { paragraphs, phrases, total: before }
  }, [text])

  // Миллисекунд на букву: обычная скорость, а для короткого текста — помедленнее, чтобы волна была видна.
  const perChar = Math.max(1000 / SPEED, Math.min(1000 / MIN_SPEED, MIN_TOTAL / Math.max(1, total))) * pace

  useEffect(() => {
    const element = root.current
    if (!element) return
    const spans = element.querySelectorAll<HTMLElement>(".ph")
    let timer = 0
    const finish = () => {
      element.dataset.done = "true"
      done.current?.()
    }
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return finish()

    // Фраза выходит, когда волна дошла до её начала. По часам, а не по кадрам: если вкладку свернули,
    // к возвращению текст уже на месте.
    let frame = 0
    let out = 0
    const started = performance.now()
    const tick = (now: number) => {
      while (out < spans.length && now - started >= phrases[out].before * perChar) {
        spans[out].dataset.on = "true"
        out++
      }
      if (out >= spans.length) {
        // Последняя фраза ещё проявляется: конец — когда волна прошла её всю и последнее слово набрало цвет.
        const last = phrases[phrases.length - 1]
        timer = window.setTimeout(finish, Math.max(0, (last.before + last.chars) * perChar - (now - started)) + FADE)
        return
      }
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => {
      cancelAnimationFrame(frame)
      window.clearTimeout(timer)
    }
  }, [phrases, perChar])

  return (
    <div className="typed" ref={root}>
      {paragraphs.map((block, position) => (
        <p key={position}>
          {block.map((phrase, inner) => (
            <span key={inner} className="ph">
              {phrase.map(({ word, offset }, at) => (
                <span key={at}>
                  {/* Слово ждёт, пока до него дойдёт волна; до этого оно стоит бледным вместе со всей фразой. */}
                  <span className="w" style={{ animationDelay: `${Math.round(offset * perChar)}ms` }}>
                    {word}
                  </span>{" "}
                </span>
              ))}
            </span>
          ))}
        </p>
      ))}
    </div>
  )
}
