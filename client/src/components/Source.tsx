import { useEffect, useState } from "react"
import { ExternalLink, X } from "lucide-react"
import { getSource } from "../api"
import type { Citation, Source } from "../api"
import { Sheet, usePresence } from "./Sheet"

// Раздел статьи не меняется, поэтому повторное открытие берётся из памяти.
const cache = new Map<string, Source>()

function useSource(citation: Citation | null) {
  const key = citation ? `${citation.article_id}/${citation.section_id}` : null
  const [state, setState] = useState<{ key: string | null; source: Source | null; failed: boolean }>({ key: null, source: null, failed: false })
  const [attempt, setAttempt] = useState(0)
  useEffect(() => {
    if (!citation || !key) return
    const ready = cache.get(key)
    if (ready) {
      setState({ key, source: ready, failed: false })
      return
    }
    let alive = true
    setState({ key, source: null, failed: false })
    getSource(citation.article_id, citation.section_id)
      .then((source) => {
        cache.set(key, source)
        if (alive) setState({ key, source, failed: false })
      })
      .catch(() => alive && setState({ key, source: null, failed: true }))
    return () => {
      alive = false
    }
  }, [key, attempt]) // eslint-disable-line react-hooks/exhaustive-deps
  const current = state.key === key ? state : { key, source: null, failed: false }
  return { ...current, retry: () => setAttempt((value) => value + 1) }
}

function SourceBody({ citation }: { citation: Citation }) {
  const { source, failed, retry } = useSource(citation)
  const doi = source?.doi ?? citation.doi
  return (
    <div className="source">
      <h3 lang="en">{citation.article_title}</h3>
      <div className="source-meta">
        <span>
          Раздел: <span lang="en">{citation.section_title}</span>
        </span>
        {citation.year ? <span>{citation.year}</span> : null}
      </div>
      {source ? (
        <p className="source-text rise" lang="en">
          {source.text}
        </p>
      ) : failed ? (
        <div className="source-failed" role="alert">
          <span>Не удалось загрузить текст раздела.</span>
          <button type="button" className="button-ghost" onClick={retry}>
            Повторить
          </button>
        </div>
      ) : (
        <div className="skeleton" role="status" aria-label="Загружаю раздел статьи">
          <i />
          <i />
          <i />
          <i />
        </div>
      )}
      <span className="note">Текст раздела приведён на языке статьи.</span>
      {doi && (
        <a className="button-outline source-link" href={`https://doi.org/${doi}`} target="_blank" rel="noreferrer">
          Открыть статью · doi.org <ExternalLink size={16} aria-hidden="true" />
        </a>
      )}
    </div>
  )
}

type Props = { citation: Citation | null; wide: boolean; onClose: () => void }

/** Источник ответа: на широком экране — панель справа от разговора, на узком — лист снизу. */
export function SourceView({ citation, wide, onClose }: Props) {
  // Пока панель уезжает, показываем то, что в ней было.
  const [last, setLast] = useState<Citation | null>(citation)
  useEffect(() => {
    if (citation) setLast(citation)
  }, [citation])
  const { mounted, visible } = usePresence(!!citation && wide, 320)
  const shown = citation ?? last

  useEffect(() => {
    if (!citation || !wide) return
    const onKey = (event: KeyboardEvent) => event.key === "Escape" && onClose()
    document.addEventListener("keydown", onKey)
    return () => document.removeEventListener("keydown", onKey)
  }, [citation, wide, onClose])

  if (!wide) {
    return (
      <Sheet open={!!citation} onClose={onClose} title="Источник ответа">
        {shown && <SourceBody citation={shown} />}
      </Sheet>
    )
  }
  if (!mounted || !shown) return null
  return (
    <aside className="side" data-open={visible} aria-label="Источник ответа">
      <div className="side-inner">
        <div className="sheet-head">
          <h2>Источник ответа</h2>
          <button type="button" className="icon-button" aria-label="Закрыть источник" onClick={onClose}>
            <X size={20} />
          </button>
        </div>
        <SourceBody citation={shown} />
      </div>
    </aside>
  )
}
