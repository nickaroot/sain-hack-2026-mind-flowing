// Заглушки вместо стенда: на них собираются и проверяются экраны, лимит нейросети не тратится.
// Список экспертов и источник — настоящие ответы стенда от 19.11. Тексты ответов про кофе начинаются
// как настоящие (см. client/README.md), продолжения дописаны для макета.
//
// Сценарии по словам в вопросе (первая попытка падает, повтор проходит):
//   «мелатонин», «доза», «диагноз», «лечение» → отказ эксперта (refused: true)
//   «429» → эксперты заняты · «лимит» → дневной запас кончился · «503» → сервер занят
//   «504» → ответ не успел · «запасной» → degraded: true · «сеть» → нет связи
// В адресе: ?delay=1500 — задержка ответа в мс, ?extra — четвёртый эксперт, которого клиент не знает.
import { ApiError } from "./api"
import type { ChatResponse, Expert, ExpertsResponse, Source } from "./api"

const params = new URLSearchParams(location.search)
const DELAY = Number(params.get("delay")) || 6000
const EXTRA = params.has("extra")
const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

const EXPERTS: Expert[] = [
  {
    expert_id: "clinician",
    name: "Сомнолог-клиницист",
    description:
      "Думает как врач, который читает исследования: сначала механизм, затем сила доказательств (какой тип исследования, сколько участников, кто они), затем граница применимости: к кому вывод относится и к кому нет, что исследование показало и чего не показало. Отделяет связь от причины и называет вероятности, а не гарантии. Личных планов по шагам не строит: даёт вывод и условие, при котором он верен. В отказе объясняет, почему вопрос требует очного осмотра и какие сведения для него нужны, и оставляет пользователю один безопасный вывод из базы.",
    speech_style:
      "Спокойные полные предложения, ровный темп, без восклицаний, сленга и шуток. Начинает с вывода, заканчивает оговоркой о границе. Каждый термин сразу поясняет. Не говорит «точно», «всегда», «гарантированно».",
  },
  {
    expert_id: "biohacker",
    name: "Биохакер-экспериментатор",
    description:
      "Думает как исследователь собственного сна: превращает вывод статьи в личную проверку. Формулирует гипотезу, выбирает одну переменную, которую пользователь меняет, метрику и способ её снять (дневник, будильник, браслет), срок проверки и критерий, по которому станет ясно, сработало ли; заранее говорит, что делать при отрицательном результате. Данные браслета считает сигналом для проверки, а не диагнозом. Веществ, доз и лечения не предлагает даже как эксперимент. В отказе переводит запретную просьбу в то, что можно безопасно измерить и поменять самому.",
    speech_style:
      "Короткие энергичные фразы, деловой темп. Мыслит шагами и сроками. Числа из статьи называет только как условия исследования или порог для проверки, а не как обещание. Без медицинских терминов, где можно сказать проще.",
  },
  {
    expert_id: "nutritionist",
    name: "Нутрициолог здравого смысла",
    description:
      "Думает о том, где вывод статьи встречается с обычным днём пользователя: едой, напитками, распорядком, семьёй и работой. Ищет одну маленькую замену, а не запрет, и привязывает её к привычке, которая уже есть (после ужина, по дороге домой, перед душем). Переводит числа исследования в бытовые меры (чашки, часы, ужин пораньше) и говорит, как не бросить через три дня. Добавок, лечебных диет и дозировок не назначает. В отказе предлагает бытовую замену той потребности, которая стоит за просьбой.",
    speech_style:
      "Тёплый неторопливый тон, простые слова, бытовые примеры. Не запугивает и не требует идеала. Один шаг за раз, без списков из десяти пунктов.",
  },
]

const COACH: Expert = {
  expert_id: "coach",
  name: "Тренер режима",
  description:
    "Думает расписанием: превращает вывод статьи в понятное время подъёма и отбоя. Проверяет, насколько режим выполним на этой неделе, и сдвигает его маленькими шагами. Лечения и доз не назначает.",
  speech_style: "Чёткие дружелюбные фразы. Говорит про время и порядок действий. Хвалит за сделанное.",
}

const CITATION = {
  article_id: "art_20",
  section_id: "s001",
  article_title: "Dose and timing effects of caffeine on subsequent sleep: a randomized clinical crossover trial",
  section_title: "Abstract › Study Objectives",
  year: 2025,
  doi: "10.1093/sleep/zsae230",
}

const ANSWERS: Record<string, string> = {
  clinician:
    "Да, чашка кофе после обеда может затруднить засыпание вечером. В рандомизированном перекрёстном исследовании сравнивали обычную и высокую дозу кофеина, принятую утром, днём и вечером, и оценивали сон следующей ночью.\n\nМеханизм понятен: кофеин блокирует рецепторы аденозина — вещества, которое накапливается за день и вызывает сонливость. Граница вывода: это средний эффект у участников исследования, а личная чувствительность к кофеину заметно различается.",
  biohacker:
    "Да, кофе после обеда может сдвинуть засыпание. В рандомизированном перекрёстном исследовании сравнивали дозу и время приёма — значит, время последней чашки стоит проверить на себе.\n\nГипотеза: без послеобеденного кофе ты засыпаешь быстрее. Меняй одно — время последней чашки. Метрика — минуты до засыпания, по дневнику. Срок — две недели: одна как обычно, одна без кофе после обеда. Разницы нет — значит, это не твой рычаг, ищем другой.",
  nutritionist:
    "Да, чашка кофе после обеда может мешать заснуть вечером: исследователи сравнивали кофе утром, днём и вечером и смотрели, как после этого спится.\n\nЗапрещать себе кофе не нужно. Попробуйте одну замену: послеобеденную чашку поменяйте на напиток без кофеина, а любимый кофе оставьте на утро. Привяжите это к тому, что уже есть: налили чай сразу после обеда — и ритуал на месте.",
  coach:
    "Да, кофе после обеда может отодвинуть засыпание. Поставьте себе простую границу: последняя чашка — до обеда. Продержитесь неделю и отметьте, во сколько удалось уснуть.",
}

const REFUSALS: Record<string, string> = {
  clinician:
    "Дозу или лечение я не назову: это назначение, а для него нужен очный приём — врач учитывает возраст, другие лекарства и причину нарушений сна. Безопасный вывод из базы такой: на засыпание влияет то, что вы делаете днём, например время последней чашки кофе. Об этом могу рассказать подробно.",
  biohacker:
    "Дозировки — не моя территория, с этим к врачу. А вот что можно проверить самому: время последней чашки кофе. Одна переменная, дневник засыпания, две недели — и у тебя будут свои данные вместо чужих советов.",
  nutritionist:
    "Здесь я вам не помощник: добавки и дозы назначает врач. Давайте начнём с простого и безопасного — посмотрим, что вы пьёте после обеда. Одна замена напитка часто даёт больше, чем кажется.",
  coach: "С дозами и лечением — к врачу. А режим мы можем поправить сами: начнём с времени последней чашки кофе.",
}

/** Счётчик попыток по тексту вопроса: сбой случается только в первый раз. */
const attempts = new Map<string, number>()

export async function experts(): Promise<ExpertsResponse> {
  await wait(350)
  return { experts: EXTRA ? [...EXPERTS, COACH] : EXPERTS, user_id: null }
}

export async function chat(expertId: string, text: string): Promise<ChatResponse> {
  const lower = text.toLowerCase()
  const attempt = (attempts.get(lower) ?? 0) + 1
  attempts.set(lower, attempt)
  if (attempt === 1) {
    if (lower.includes("сеть")) {
      await wait(800)
      throw new ApiError("network", "Failed to fetch")
    }
    await wait(1200)
    if (lower.includes("429")) throw new ApiError("busy", "Эксперты заняты другими вопросами, повторите через несколько секунд", 429, 8)
    if (lower.includes("лимит")) throw new ApiError("quota", "На сегодня вопросы закончились, приходите завтра", 429, 3600)
    if (lower.includes("503")) throw new ApiError("paused", "Сервер готовится к прогону, попробуйте через минуту", 503, 60)
    if (lower.includes("504")) throw new ApiError("timeout", "Ответ не успел за 90 секунд", 504)
  }
  await wait(DELAY)
  const refused = /мелатонин|доз|диагноз|лечени|таблет/.test(lower)
  const degraded = lower.includes("запасной")
  const texts = refused ? REFUSALS : ANSWERS
  const ids = (EXTRA ? [...EXPERTS, COACH] : EXPERTS).map((expert) => expert.expert_id)
  const answers = Object.fromEntries(ids.map((id) => [id, { text: texts[id] ?? ANSWERS.clinician, refused }]))
  const own = degraded
    ? { text: "Сейчас не получилось подготовить полный ответ. Загляните в раздел статьи по ссылке и задайте вопрос ещё раз.", refused: false }
    : answers[expertId] ?? answers.clinician
  if (degraded) answers[expertId] = own
  return { expert_id: expertId, text: own.text, refused: own.refused, degraded, answers, citations: [CITATION] }
}

export async function source(articleId: string, sectionId: string): Promise<Source> {
  await wait(700)
  if (articleId !== CITATION.article_id || sectionId !== CITATION.section_id) {
    throw new ApiError("unknown", "Раздел не найден", 404)
  }
  return {
    ...CITATION,
    text: "To investigate the effect of a typical dose of caffeine and a high dose of caffeine consumed in the morning, afternoon, and evening on subsequent sleep.",
  }
}

export async function transcribe(): Promise<string> {
  await wait(1200)
  return "Мешает ли чашка кофе после обеда уснуть вечером?"
}

/** ?tone — вместо синтезатора браузера вернуть звуковой файл: так проверяется путь настоящего голоса. */
export async function speak(): Promise<Blob | null> {
  await wait(900)
  if (!params.has("tone")) return null
  const rate = 16000
  const seconds = 4
  const samples = new Int16Array(rate * seconds)
  for (let index = 0; index < samples.length; index++) {
    const time = index / rate
    // «Слоги»: тон, громкость которого то нарастает, то пропадает.
    const syllable = Math.max(0, Math.sin(time * Math.PI * 4.2)) * (Math.sin(time * 2.1) > -0.6 ? 1 : 0)
    samples[index] = Math.sin(time * 2 * Math.PI * 180) * syllable * 9000
  }
  const header = new DataView(new ArrayBuffer(44))
  const text = (offset: number, value: string) => [...value].forEach((char, index) => header.setUint8(offset + index, char.charCodeAt(0)))
  text(0, "RIFF")
  header.setUint32(4, 36 + samples.byteLength, true)
  text(8, "WAVEfmt ")
  header.setUint32(16, 16, true)
  header.setUint16(20, 1, true)
  header.setUint16(22, 1, true)
  header.setUint32(24, rate, true)
  header.setUint32(28, rate * 2, true)
  header.setUint16(32, 2, true)
  header.setUint16(34, 16, true)
  text(36, "data")
  header.setUint32(40, samples.byteLength, true)
  return new Blob([header, samples], { type: "audio/wav" })
}
