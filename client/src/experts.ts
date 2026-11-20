// Как эксперт выглядит в интерфейсе. Сам список приходит с сервера (GET /api/experts); здесь — только
// короткие подписи для трёх известных экспертов. Незнакомый эксперт получает подписи из своих
// description и speech_style, цвет — из идентификатора, значок — первую букву имени: править клиент не нужно.
import { Activity, Leaf, Stethoscope } from "lucide-react"
import type { LucideIcon } from "lucide-react"
import type { CSSProperties } from "react"
import type { Expert } from "./api"

export type Persona = {
  id: string
  name: string
  /** Короткое имя для переключателя и поля ввода: первое слово названия. */
  short: string
  tagline: string
  traits: string[]
  /** Реплика, которую эксперт «произносит» в галерее. null — показываем, как он думает. */
  sample: string | null
  about: string
  speechStyle: string
  /** Оттенок акцента в пространстве OKLCH; яркость и насыщенность задаёт тема. */
  hue: number
  icon: LucideIcon | null
  /** Множитель темпа появления текста: манера речи видна и в движении. */
  pace: number
}

/** Один вопрос для всех: различие экспертов — в подаче, а не в подписях (требование кейса). */
export const SAMPLE_QUESTION = "Мешает ли кофе после обеда уснуть?"

type Known = Pick<Persona, "tagline" | "traits" | "sample" | "hue" | "icon" | "pace">

// Реплики-образцы написаны по примерам настоящих ответов стенда; без доз, диагнозов и назначений.
const KNOWN: Record<string, Known> = {
  clinician: {
    tagline: "Объясняет механизм, силу доказательств и границу вывода",
    traits: ["Начинает с вывода", "Вероятности вместо гарантий", "Заканчивает оговоркой"],
    sample:
      "Да, может затруднить засыпание. Кофеин блокирует аденозин — сигнал сонливости; в перекрёстном исследовании сравнивали разные дозы и время приёма. Это средний результат: чувствительность у людей разная.",
    hue: 168,
    icon: Stethoscope,
    pace: 1,
  },
  biohacker: {
    tagline: "Превращает вывод статьи в личную проверку",
    traits: ["Гипотеза → метрика → срок", "Короткие фразы", "План на случай «не сработало»"],
    sample:
      "Может. Проверь на себе: одна переменная — время последней чашки. Неделя как обычно, неделя без кофе после обеда. Метрика — минуты до засыпания в дневнике. Нет разницы — ищем другой рычаг.",
    hue: 300,
    icon: Activity,
    pace: 0.7,
  },
  nutritionist: {
    tagline: "Ищет одну бытовую замену вместо запрета",
    traits: ["Тёплый неторопливый тон", "Чашки и часы вместо чисел", "Один шаг за раз"],
    sample:
      "Может мешать, да. Только не надо запретов: оставьте сам ритуал, а после обеда наливайте что-нибудь без кофеина. Любимый кофе пусть останется на утро — так проще не бросить через три дня.",
    hue: 55,
    icon: Leaf,
    pace: 1.3,
  },
}

const sentences = (text: string) =>
  text
    .split(/(?<=[.!?…])\s+/)
    .map((part) => part.trim())
    .filter(Boolean)

const clip = (text: string, max: number) => {
  const clean = text.replace(/[.:;,\s]+$/, "")
  return clean.length <= max ? clean : `${clean.slice(0, max - 1).replace(/[\s,;:]+\S*$/, "")}…`
}

function hueFrom(id: string): number {
  let hash = 0
  for (const char of id) hash = (hash * 31 + char.charCodeAt(0)) % 360
  return hash
}

export function persona(expert: Expert): Persona {
  const known = KNOWN[expert.expert_id]
  const base = {
    id: expert.expert_id,
    name: expert.name,
    short: expert.name.split(/[\s-]/)[0] || expert.name,
    about: expert.description,
    speechStyle: expert.speech_style,
  }
  if (known) return { ...base, ...known }
  return {
    ...base,
    tagline: clip(sentences(expert.description)[0] ?? expert.name, 90),
    traits: sentences(expert.speech_style)
      .slice(0, 3)
      .map((part) => clip(part, 38)),
    sample: null,
    hue: hueFrom(expert.expert_id),
    icon: null,
    pace: 1,
  }
}

/** Что эксперт произносит в галерее: реплика-образец, а у незнакомого эксперта — начало его описания. */
export const galleryLine = (persona: Persona) => persona.sample ?? sentences(persona.about).slice(0, 2).join(" ")

/** Акцент эксперта задаётся одним числом — оттенком; остальное считает CSS (см. styles.css). */
export const accentStyle = (hue: number) => ({ "--h": String(hue) }) as CSSProperties
