// Слот для анимированных аватаров. Пока здесь пусто — везде рисуется плейсхолдер (components/Avatar.tsx).
//
// Чтобы подключить свой аватар, положите компонент рядом и впишите его сюда по expert_id:
//
//   import { ClinicianAvatar } from "./ClinicianAvatar"
//   export const customAvatars = { clinician: ClinicianAvatar }
//
// Компонент получает AvatarProps: размер в пикселях и состояние. Состояний пять:
//   idle      — эксперт не выбран (сосед в галерее)
//   active    — выбран и молчит
//   speaking  — говорит: реплика в галерее или озвучивание ответа
//   thinking  — готовит ответ (ожидание сервера, 10–20 секунд)
//   listening — пользователь записывает вопрос голосом
// Внутри может быть что угодно: Lottie, Rive, видео, SVG. Эксперты вымышленные — без сходства с живыми людьми.
import type { ComponentType } from "react"

export type AvatarState = "idle" | "active" | "speaking" | "thinking" | "listening"
export type AvatarProps = { expertId: string; name: string; state: AvatarState; size: number }

export const customAvatars: Record<string, ComponentType<AvatarProps>> = {}
