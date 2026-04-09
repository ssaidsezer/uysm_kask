import { useQuery } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { api } from '../api/client'
import type { ModelProfile } from '../pages/ModelProfilesPage'
import { ragTypeToFlags, type RagTypeUi } from '../utils/collections'
import { ragModeToApi, type RagModeApi, type RagModeUiLabel } from '../eval/evalRagMode'

export type EvalBackendChoice = 'OpenAI' | 'Yerel (Ollama)'

export function useEvalPageBasics() {
  const ollamaQ = useQuery({
    queryKey: ['ollama-models'],
    queryFn: async () => (await api.get('/api/models/ollama')).data,
  })
  const embedQ = useQuery({
    queryKey: ['embed-models'],
    queryFn: async () => (await api.get('/api/models/embeddings')).data,
  })
  const cfg = useQuery({
    queryKey: ['config'],
    queryFn: async () => (await api.get('/api/config')).data,
  })
  const profilesQ = useQuery({
    queryKey: ['model-profiles'],
    queryFn: async () => (await api.get<{ profiles: ModelProfile[] }>('/api/model-profiles')).data,
  })

  const allModels = ollamaQ.data?.models ?? []
  const embedModels = embedQ.data?.models ?? []

  const [qaSelected, setQaSelected] = useState<string[]>([])
  const [customModels, setCustomModels] = useState<string[]>([])

  const [evalEnabled, setEvalEnabled] = useState(true)
  const [evalBackend, setEvalBackend] = useState<EvalBackendChoice>('OpenAI')
  const [evalModelName, setEvalModelName] = useState('')
  const [localEvalModel, setLocalEvalModel] = useState('')

  const [embedModel, setEmbedModel] = useState('')
  const [ragType, setRagType] = useState<RagTypeUi>('Klasik')
  const [collectionName, setCollectionName] = useState('')

  const [ragModeUi, setRagModeUi] = useState<RagModeUiLabel>("RAG'li")
  const [k, setK] = useState(5)
  const [scoreTh, setScoreTh] = useState(0.55)

  const [openaiKey, setOpenaiKey] = useState('')

  const [useSavedQaDefaults, setUseSavedQaDefaults] = useState(true)
  const [bulkQaProfileId, setBulkQaProfileId] = useState('')
  const [useSavedEvalDefaults, setUseSavedEvalDefaults] = useState(true)
  const [evalProfileId, setEvalProfileId] = useState('')
  const [thinkingEnabled, setThinkingEnabled] = useState(false)

  useEffect(() => {
    if (cfg.data?.eval_model_name) setEvalModelName(cfg.data.eval_model_name)
  }, [cfg.data])

  useEffect(() => {
    if (embedModels.length && !embedModel) setEmbedModel(embedModels[0])
  }, [embedModels, embedModel])

  useEffect(() => {
    if (allModels.length && !localEvalModel) setLocalEvalModel(allModels[0])
  }, [allModels, localEvalModel])

  const { smartRag, retrievalMode } = ragTypeToFlags(ragType)
  const ragModeApi: RagModeApi = ragModeToApi[ragModeUi]

  const qaProfiles = profilesQ.data?.profiles ?? []
  const evalProfiles = profilesQ.data?.profiles ?? []

  return {
    ollamaQ,
    embedQ,
    cfg,
    profilesQ,
    allModels,
    embedModels,
    qaSelected,
    setQaSelected,
    customModels,
    setCustomModels,
    evalEnabled,
    setEvalEnabled,
    evalBackend,
    setEvalBackend,
    evalModelName,
    setEvalModelName,
    localEvalModel,
    setLocalEvalModel,
    embedModel,
    setEmbedModel,
    ragType,
    setRagType,
    collectionName,
    setCollectionName,
    ragModeUi,
    setRagModeUi,
    k,
    setK,
    scoreTh,
    setScoreTh,
    openaiKey,
    setOpenaiKey,
    useSavedQaDefaults,
    setUseSavedQaDefaults,
    bulkQaProfileId,
    setBulkQaProfileId,
    useSavedEvalDefaults,
    setUseSavedEvalDefaults,
    evalProfileId,
    setEvalProfileId,
    thinkingEnabled,
    setThinkingEnabled,
    smartRag,
    retrievalMode,
    ragModeApi,
    qaProfiles,
    evalProfiles,
  }
}
