/** Shared RAG mode labels for manual chat + CSV eval UIs (must match backend `rag` | `no_rag` | `both`). */

export const RAG_MODE_UI = ["RAG'li", "RAG'siz", 'İkisi birden'] as const

export type RagModeUiLabel = (typeof RAG_MODE_UI)[number]

export type RagModeApi = 'rag' | 'no_rag' | 'both'

export const ragModeToApi: Record<RagModeUiLabel, RagModeApi> = {
  "RAG'li": 'rag',
  "RAG'siz": 'no_rag',
  'İkisi birden': 'both',
}

/** Build `qa_profile_by_model` when a single profile is applied to all selected QA models. */
export function buildQaProfileByModel(
  bulkQaProfileId: string,
  qaSelected: string[],
): Record<string, string> {
  const qaByModel: Record<string, string> = {}
  if (bulkQaProfileId && qaSelected.length) {
    for (const m of qaSelected) qaByModel[m] = bulkQaProfileId
  }
  return qaByModel
}
