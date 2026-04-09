import { useMutation, useQuery } from '@tanstack/react-query'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Collapse from '@mui/material/Collapse'
import CircularProgress from '@mui/material/CircularProgress'
import Divider from '@mui/material/Divider'
import FormControlLabel from '@mui/material/FormControlLabel'
import Paper from '@mui/material/Paper'
import Radio from '@mui/material/Radio'
import RadioGroup from '@mui/material/RadioGroup'
import Slider from '@mui/material/Slider'
import Stack from '@mui/material/Stack'
import Switch from '@mui/material/Switch'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { useEffect, useState } from 'react'
import { api } from '../api/client'
import { CollectionPicker } from '../components/CollectionPicker'
import { QaModelPicker } from '../components/QaModelPicker'
import { ragTypeToFlags, type RagTypeUi } from '../utils/collections'

const RAG_MODE_UI = ["RAG'li", "RAG'siz", 'İkisi birden'] as const
const ragModeToApi: Record<(typeof RAG_MODE_UI)[number], 'rag' | 'no_rag' | 'both'> = {
  "RAG'li": 'rag',
  "RAG'siz": 'no_rag',
  'İkisi birden': 'both',
}

function chunkTitle(retrievalMode: string, smartRag: boolean) {
  if (retrievalMode === 'bm25' && smartRag)
    return 'Retrieved Chunks — BM25 Smart (child arama, parent bağlam)'
  if (retrievalMode === 'bm25') return 'Retrieved Chunks — BM25 Klasik'
  if (smartRag) return 'Retrieved Chunks — Smart RAG'
  return 'Retrieved Chunks'
}

export function ChatEvalPage() {
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

  const allModels = ollamaQ.data?.models ?? []
  const embedModels = embedQ.data?.models ?? []

  const [qaSelected, setQaSelected] = useState<string[]>([])
  const [customModels, setCustomModels] = useState<string[]>([])

  const [evalEnabled, setEvalEnabled] = useState(true)
  const [evalBackend, setEvalBackend] = useState<'OpenAI' | 'Yerel (Ollama)'>('OpenAI')
  const [evalModelName, setEvalModelName] = useState('')
  const [localEvalModel, setLocalEvalModel] = useState('')

  const [embedModel, setEmbedModel] = useState('')
  const [ragType, setRagType] = useState<RagTypeUi>('Klasik')
  const [collectionName, setCollectionName] = useState('')

  const [ragModeUi, setRagModeUi] = useState<(typeof RAG_MODE_UI)[number]>("RAG'li")
  const [k, setK] = useState(5)
  const [scoreTh, setScoreTh] = useState(0.55)
  const [think, setThink] = useState(false)

  const [question, setQuestion] = useState('')
  const [expected, setExpected] = useState('')
  const [openaiKey, setOpenaiKey] = useState('')
  const [formError, setFormError] = useState<string | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)

  const [resp, setResp] = useState<any>(null)

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
  const ragModeApi = ragModeToApi[ragModeUi]

  const evalMut = useMutation({
    mutationFn: async () => {
      const { data } = await api.post('/api/chat/eval', {
        question: question.trim(),
        expected_answer: expected,
        rag_mode: ragModeApi,
        k,
        qa_models_selected: qaSelected,
        eval_enabled: evalEnabled,
        eval_backend: evalBackend,
        eval_model_name: evalModelName,
        local_eval_model_name: localEvalModel || null,
        openai_api_key: openaiKey || null,
        collection_name: collectionName,
        embed_model: embedModel || null,
        think,
        smart_rag: smartRag,
        score_threshold: scoreTh,
        retrieval_mode: retrievalMode,
      })
      return data
    },
    onMutate: () => {
      setIsGenerating(true)
      setFormError(null)
      setResp(null)
    },
    onSuccess: (data) => setResp(data),
    onError: (err: any) => {
      const apiMessage =
        err?.response?.data?.detail ||
        err?.response?.data?.message ||
        err?.message ||
        'Yanıtlar üretilirken bir hata oluştu.'
      setFormError(String(apiMessage))
    },
    onSettled: () => setIsGenerating(false),
  })

  const downloadChatCsv = () => {
    if (!resp?.chat_eval_rows?.length) return
    const rows = resp.chat_eval_rows as Record<string, unknown>[]
    const keys = [...new Set(rows.flatMap((r) => Object.keys(r)))]
    const esc = (v: unknown) => {
      const s = v == null ? '' : String(v)
      if (s.includes('"') || s.includes(',') || s.includes('\n')) return `"${s.replace(/"/g, '""')}"`
      return s
    }
    const lines = [keys.join(','), ...rows.map((r) => keys.map((k) => esc(r[k])).join(','))]
    const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'chat_results.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <Box>
      {ollamaQ.data?.error && <Alert severity="error">{ollamaQ.data.error}</Alert>}
      <QaModelPicker
        allModels={allModels}
        filteredEmbeddingCount={ollamaQ.data?.filtered_embedding_count ?? 0}
        selected={qaSelected}
        onChange={setQaSelected}
        customModels={customModels}
        onCustomModelsChange={setCustomModels}
      />

      <Box sx={{ mb: 2 }}>
        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: { xs: '1fr', lg: 'repeat(3, minmax(0, 1fr))' },
            gap: 2,
          }}
        >
          <Box sx={{ p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
            <Typography variant="subtitle2" sx={{ mb: 1 }}>
              Değerlendirme
            </Typography>
            <FormControlLabel
              control={<Switch checked={evalEnabled} onChange={(_, v) => setEvalEnabled(v)} />}
              label="Değerlendir"
            />
            {evalEnabled && (
              <Stack spacing={1} sx={{ mt: 1 }}>
                <TextField
                  select
                  label="Motor"
                  value={evalBackend}
                  onChange={(e) =>
                    setEvalBackend(e.target.value as 'OpenAI' | 'Yerel (Ollama)')
                  }
                  slotProps={{ select: { native: true } }}
                  size="small"
                >
                  <option>OpenAI</option>
                  <option>Yerel (Ollama)</option>
                </TextField>
                {evalBackend === 'OpenAI' ? (
                  <>
                    <TextField
                      size="small"
                      label="OpenAI model"
                      value={evalModelName}
                      onChange={(e) => setEvalModelName(e.target.value)}
                    />
                    <TextField
                      size="small"
                      label="OpenAI API key (opsiyonel)"
                      value={openaiKey}
                      onChange={(e) => setOpenaiKey(e.target.value)}
                    />
                  </>
                ) : (
                  <TextField
                    select
                    size="small"
                    label="Yerel eval"
                    value={localEvalModel}
                    onChange={(e) => setLocalEvalModel(e.target.value)}
                    slotProps={{ select: { native: true } }}
                  >
                    {allModels.map((m: string) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </TextField>
                )}
              </Stack>
            )}
          </Box>
          <Box sx={{ p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
            <Typography variant="subtitle2" sx={{ mb: 1 }}>
              Embedding ve retrieval
            </Typography>
            <CollectionPicker
              embedModels={embedModels}
              embedLabel="Embedding modeli"
              ragTypeLabel="Retrieval tipi"
              ragHelp="Smart parent/child; BM25 anahtar kelime."
              horizontal={false}
              classicLabel="Klasik koleksiyon"
              smartLabel="Smart koleksiyon"
              embedModel={embedModel}
              onEmbedModel={setEmbedModel}
              ragType={ragType}
              onRagType={setRagType}
              collectionName={collectionName}
              onCollectionName={setCollectionName}
            />
          </Box>
          <Box sx={{ p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
            <Typography variant="subtitle2" sx={{ mb: 1 }}>
              RAG ayarları
            </Typography>
            <RadioGroup
              value={ragModeUi}
              onChange={(_, v) => setRagModeUi(v as (typeof RAG_MODE_UI)[number])}
            >
              {RAG_MODE_UI.map((r) => (
                <FormControlLabel key={r} value={r} control={<Radio />} label={r} />
              ))}
            </RadioGroup>
            <Stack direction="row" spacing={2} sx={{ mt: 1, alignItems: 'center', flexWrap: 'wrap' }}>
              {ragModeApi !== 'no_rag' && (
                <TextField
                  type="number"
                  label="k"
                  value={k}
                  onChange={(e) => setK(+e.target.value)}
                  size="small"
                  sx={{ width: 120 }}
                />
              )}
              <FormControlLabel
                control={<Switch checked={think} onChange={(_, v) => setThink(v)} />}
                label="Thinking"
              />
            </Stack>
            {ragModeApi !== 'no_rag' && retrievalMode === 'vector' && (
              <Box sx={{ mt: 1 }}>
                <Typography variant="caption">Score threshold: {scoreTh}</Typography>
                <Slider
                  min={0.1}
                  max={1}
                  step={0.05}
                  value={scoreTh}
                  onChange={(_, v) => setScoreTh(v as number)}
                />
              </Box>
            )}
          </Box>
        </Box>
      </Box>

      <Box sx={{ mb: 2 }}>
        <Typography variant="subtitle2" sx={{ mb: 1 }}>
          Soru ve referans
        </Typography>
        <Stack direction={{ xs: 'column', md: 'row' }} spacing={2}>
          <TextField
            label="Soru"
            multiline
            minRows={6}
            fullWidth
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
          />
          <TextField
            label="Beklenen / referans cevap"
            multiline
            minRows={6}
            fullWidth
            value={expected}
            onChange={(e) => setExpected(e.target.value)}
          />
        </Stack>
      </Box>

      <Button
        variant="contained"
        fullWidth
        disabled={isGenerating}
        onClick={() => {
          const q = question.trim()
          if (!q) {
            setFormError('Lütfen bir soru gir.')
            return
          }
          if (ragModeApi !== 'no_rag' && !collectionName.trim()) {
            setFormError('Seçili retrieval tipi için geçerli bir koleksiyon bulunamadı.')
            return
          }
          evalMut.mutate()
        }}
      >
        {isGenerating ? 'Yanıtlar üretiliyor...' : 'Soruyu değerlendir'}
      </Button>

      {isGenerating && (
        <Alert severity="info" sx={{ mt: 2 }}>
          <Stack direction="row" spacing={1} sx={{ alignItems: 'center' }}>
            <CircularProgress size={16} />
            <Typography variant="body2">
              Yanıtlar üretiliyor. İşlem bitince buton otomatik açılacak.
            </Typography>
          </Stack>
        </Alert>
      )}

      {formError && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {formError}
        </Alert>
      )}

      {resp?.errors?.length > 0 &&
        resp.errors.map((e: string, i: number) => (
          <Alert key={i} severity="error" sx={{ mt: 2 }}>
            {e}
          </Alert>
        ))}

      {resp && (
        <Box sx={{ mt: 3 }}>
          {resp.chunk_warning && <Alert severity="info">{resp.chunk_warning}</Alert>}
          {resp.chunks?.length > 0 && (
            <>
              <Typography variant="subtitle1" sx={{ mt: 2 }}>
                {chunkTitle(retrievalMode, smartRag)}
              </Typography>
              <Box
                sx={{
                  display: 'grid',
                  gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' },
                  gap: 1,
                  mt: 1,
                }}
              >
                {resp.chunks.map((c: any, i: number) => (
                  <Paper
                    key={i}
                    variant="outlined"
                    sx={{ p: 1, borderColor: 'divider', bgcolor: 'background.paper' }}
                  >
                    <Typography variant="caption" color="primary">
                      Chunk {i + 1} — {smartRag ? 'Child / Parent' : ''}
                    </Typography>
                    {smartRag && c.child_text && (
                      <Typography
                        variant="caption"
                        sx={{ display: 'block', maxHeight: 60, overflow: 'auto', color: '#aaa' }}
                      >
                        {c.child_text}
                      </Typography>
                    )}
                    <Typography
                      variant="body2"
                      sx={{ maxHeight: 100, overflow: 'auto', mt: 0.5, fontSize: '0.78rem' }}
                    >
                      {c.text}
                    </Typography>
                    <Typography variant="caption" sx={{ float: 'right', color: '#4caf50' }}>
                      {Number(c.score).toFixed(3)}
                    </Typography>
                  </Paper>
                ))}
              </Box>
            </>
          )}

          <Divider sx={{ my: 2 }} />

          {resp.model_results?.map((mr: any) => (
            <Box key={mr.model_name} sx={{ mb: 3 }}>
              <Typography variant="h6">{mr.model_name}</Typography>
              {mr.error && <Alert severity="error">{mr.error}</Alert>}
              {ragModeApi === 'both' && !mr.error ? (
                <Stack direction={{ xs: 'column', md: 'row' }} spacing={2}>
                  <Paper sx={{ p: 2, flex: 1 }}>
                    <Typography sx={{ fontWeight: 'bold' }}>RAG&apos;li</Typography>
                    <Typography sx={{ whiteSpace: 'pre-wrap' }}>{mr.rag_answer}</Typography>
                    <Collapse in={!!mr.rag_eval && evalEnabled}>
                      <pre style={{ fontSize: 12, overflow: 'auto' }}>
                        {JSON.stringify(mr.rag_eval, null, 2)}
                      </pre>
                    </Collapse>
                  </Paper>
                  <Paper sx={{ p: 2, flex: 1 }}>
                    <Typography sx={{ fontWeight: 'bold' }}>RAG&apos;siz</Typography>
                    <Typography sx={{ whiteSpace: 'pre-wrap' }}>{mr.no_rag_answer}</Typography>
                    <Collapse in={!!mr.no_rag_eval && evalEnabled}>
                      <pre style={{ fontSize: 12, overflow: 'auto' }}>
                        {JSON.stringify(mr.no_rag_eval, null, 2)}
                      </pre>
                    </Collapse>
                  </Paper>
                </Stack>
              ) : (
                !mr.error && (
                  <>
                    <Typography sx={{ whiteSpace: 'pre-wrap', mt: 1 }}>
                      {ragModeApi === 'rag' ? mr.rag_answer : mr.no_rag_answer}
                    </Typography>
                    <Collapse in={evalEnabled}>
                      <pre style={{ fontSize: 12 }}>
                        {JSON.stringify(
                          ragModeApi === 'rag' ? mr.rag_eval : mr.no_rag_eval,
                          null,
                          2,
                        )}
                      </pre>
                    </Collapse>
                  </>
                )
              )}
              <Divider sx={{ mt: 2 }} />
            </Box>
          ))}

          {resp.chat_eval_rows?.length > 0 && (
            <Button variant="outlined" onClick={downloadChatCsv} sx={{ mt: 2 }}>
              Manuel chat sonuçlarını CSV indir
            </Button>
          )}
        </Box>
      )}
    </Box>
  )
}
