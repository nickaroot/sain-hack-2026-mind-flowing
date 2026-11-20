// Слот для аватаров. Здесь по expert_id записано, каким компонентом рисовать эксперта;
// эксперт, которого здесь нет (например, новый с сервера), получает плейсхолдер из components/Avatar.tsx.
//
// Компонент получает AvatarProps: размер в пикселях и состояние. Состояний пять:
//   idle      — эксперт не выбран (сосед в галерее)
//   active    — выбран и молчит
//   speaking  — говорит: реплика в галерее или озвучивание ответа
//   thinking  — готовит ответ (ожидание сервера, 10–20 секунд)
//   listening — пользователь записывает вопрос голосом
// getLevel отдаёт громкость звучащего голоса (0…1) или null, если звука нет: по ней движется рот.
//
// Трое экспертов — Rive-сцены (RiveAvatar, scenes.ts: сетка с костями, рот по голосу, взгляд за курсором).
// Листы кадров (SpriteAvatar, sprites.ts) остались запасным вариантом, если сцена не загрузилась.
// Эксперты вымышленные — без сходства с живыми людьми.
import type { ComponentType } from "react"
import { RiveAvatar } from "./RiveAvatar"
import { SPRITES } from "./sprites"

export type AvatarState = "idle" | "active" | "speaking" | "thinking" | "listening"
export type AvatarProps = { expertId: string; name: string; state: AvatarState; size: number; getLevel?: () => number | null }

export const customAvatars: Record<string, ComponentType<AvatarProps>> = Object.fromEntries(
  Object.keys(SPRITES).map((id) => [id, RiveAvatar]),
)
