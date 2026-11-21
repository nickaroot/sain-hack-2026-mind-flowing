// Формы рта для речи. У мультяшного рта две оси: насколько он открыт и насколько губы собраны в трубочку.
// «А» — широко открыт, «О» — открыт и округлён, «У» — почти трубочка, «И» — приоткрыт и растянут,
// «М», «Б», «П» — сомкнут. Сами формы живут в Rive-сцене (rive-lab), здесь — какая буква какую форму даёт.
//
// Сервис голоса не сообщает, когда какой звук звучит, поэтому время берём из текста: буквы реплики
// раскладываются по звучащим участкам записи (гласные длиннее согласных). Это приблизительно, но рот
// закрывается на «м» и округляется на «о» там же, где это слышно.
//
// Правила из практики мультипликации (JALI, Osipa): губы обязаны сомкнуться на «м», «б», «п»; согласные,
// которые произносит язык («л», «н», «т», «д», «к», «с»…), губами не управляют — рот на них держит форму соседних
// гласных и лишь слегка прикрывается. Иначе рот хлопает на каждой букве.

export type MouthShape = { open: number; round: number }

/** Открытие, округление (0–100) и относительная длительность звука. */
type Sound = [open: number, round: number, weight: number]

const CLOSED: Sound = [0, 8, 1]
/** Согласный языка: своей формы нет, длительность — как у прочих согласных. */
const CARRIED = 0.9
/** Насколько рот прикрывается на согласном языка относительно соседних гласных. */
const CARRY_OPEN = 0.5
const SOUNDS: Record<string, Sound> = {
  а: [100, 0, 2],
  я: [100, 0, 2],
  э: [58, 0, 1.8],
  е: [58, 0, 1.8],
  и: [32, 0, 1.6],
  ы: [36, 0, 1.6],
  о: [72, 75, 2],
  ё: [72, 75, 2],
  у: [34, 100, 1.8],
  ю: [34, 100, 1.8],
  м: CLOSED,
  б: CLOSED,
  п: CLOSED,
  ф: [10, 0, 1],
  в: [10, 0, 1],
  ш: [22, 45, 1],
  ж: [22, 45, 1],
  ч: [22, 40, 1],
  щ: [22, 45, 1],
  // Латиница встречается в названиях и терминах.
  a: [100, 0, 2],
  e: [58, 0, 1.8],
  i: [32, 0, 1.6],
  y: [32, 0, 1.4],
  o: [72, 75, 2],
  u: [34, 100, 1.8],
  w: [34, 90, 1],
  m: CLOSED,
  b: CLOSED,
  p: CLOSED,
  f: [10, 0, 1],
  v: [10, 0, 1],
}

/** Реплика как дорожка звуков: каждая буква занимает отрезок от конца предыдущей до своего `upto`. */
export type MouthTrack = { open: number; round: number; upto: number }[]

export function mouthTrack(text: string): MouthTrack {
  const sounds: (Sound | null)[] = [] // null — согласный языка: форму возьмёт у соседей
  for (const char of text.toLowerCase()) {
    if (char === "ь" || char === "ъ") continue
    if (SOUNDS[char]) sounds.push(SOUNDS[char])
    else if (/\p{L}/u.test(char)) sounds.push(null)
    else if (/\d/.test(char)) sounds.push([55, 20, 2])
    else if (/[.,!?;:—–]/.test(char)) sounds.push([0, 0, 0.6]) // на знаке препинания рот на миг смыкается
  }
  const track: MouthTrack = []
  let total = 0
  sounds.forEach((sound, index) => {
    if (sound) {
      total += sound[2]
      track.push({ open: sound[0], round: sound[1], upto: total })
      return
    }
    let before = index - 1
    while (before >= 0 && !sounds[before]) before--
    let after = index + 1
    while (after < sounds.length && !sounds[after]) after++
    const left = sounds[before] ?? sounds[after] ?? CLOSED
    const right = sounds[after] ?? sounds[before] ?? CLOSED
    total += CARRIED
    track.push({ open: 8 + (CARRY_OPEN * (left[0] + right[0])) / 2, round: (left[1] + right[1]) / 2, upto: total })
  })
  return track
}

/** Форма рта в доле `part` (0…1) от реплики. К концу звука форма плавно уходит к следующему. */
export function shapeAt(track: MouthTrack, part: number): MouthShape {
  if (!track.length) return { open: 0, round: 0 }
  const at = Math.min(0.9999, Math.max(0, part)) * track[track.length - 1].upto
  let low = 0
  let high = track.length - 1
  while (low < high) {
    const middle = (low + high) >> 1
    if (track[middle].upto > at) high = middle
    else low = middle + 1
  }
  const now = track[low]
  const next = track[Math.min(track.length - 1, low + 1)]
  const from = low ? track[low - 1].upto : 0
  const along = (at - from) / (now.upto - from)
  const mix = along < 0.6 ? 0 : (along - 0.6) / 0.4
  return { open: now.open + (next.open - now.open) * mix, round: now.round + (next.round - now.round) * mix }
}
