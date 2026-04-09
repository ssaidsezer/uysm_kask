import { useMutation, useQuery } from '@tanstack/react-query'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Checkbox from '@mui/material/Checkbox'
import FormControlLabel from '@mui/material/FormControlLabel'
import LinearProgress from '@mui/material/LinearProgress'
import Radio from '@mui/material/Radio'
import RadioGroup from '@mui/material/RadioGroup'
import Slider from '@mui/material/Slider'
import Stack from '@mui/material/Stack'
import Switch from '@mui/material/Switch'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { DataGrid } from '@mui/x-data-grid'
import type { GridColDef } from '@mui/x-data-grid'
import { useEffect, useMemo, useState } from 'react'
import { api } from '../api/client'
import { CollectionPicker } from '../components/CollectionPicker'
import { QaModelPicker } from '../components/QaModelPicker'
import { useJobPolling } from '../hooks/useJobPolling'
import { ragTypeToFlags, type RagTypeUi } from '../utils/collections'

const RAG_MODE_UI = ["RAG'li", "RAG'siz", 'İkisi birden'] as const
const ragModeToApi: Record<(typeof RAG_MODE_UI)[number], string> = {
  "RAG'li": 'rag',
  "RAG'siz": 'no_rag',
  'İkisi birden': 'both',
}

export function CsvEvalPage() {
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
  const ollamaErr = ollamaQ.data?.error

  const [qaSelected, setQaSelected] = useState<string[]>([])
  const [customModels, setCustomModels] = useState<string[]>([])

  const [evalEnabled, setEvalEnabled] = useState(true)
  const [evalBackend, setEvalBackend] = useState<'OpenAI' | 'Yerel (Ollama)'>('OpenAI')
  const [evalModelName, setEvalModelName] = useState('')
  const [localEvalModel, setLocalEvalModel] = useState('')

  const [csvFile, setCsvFile] = useState<File | null>(null)
  const [useSample, setUseSample] = useState(false)
  const [qCol, setQCol] = useState('question')
  const [aCol, setACol] = useState('answer')

  const [embedModel, setEmbedModel] = useState('')
  const [ragType, setRagType] = useState<RagTypeUi>('Klasik')
  const [collectionName, setCollectionName] = useState('')

  const [ragModeUi, setRagModeUi] = useState<(typeof RAG_MODE_UI)[number]>("RAG'li")
  const [k, setK] = useState(5)
  const [think, setThink] = useState(false)
  const [scoreTh, setScoreTh] = useState(0.55)
  const [openaiKey, setOpenaiKey] = useState('')

  const [jobId, setJobId] = useState<string | null>(null)
  const jobQ = useJobPolling(jobId)

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

  const startMut = useMutation({
    mutationFn: async () => {
      const fd = new FormData()
      if (csvFile) fd.append('csv_file', csvFile)
      fd.append('use_sample', String(useSample))
      fd.append('eval_enabled', String(evalEnabled))
      fd.append('eval_backend', evalBackend)
      fd.append('eval_model_name', evalModelName)
      if (localEvalModel) fd.append('local_eval_model_name', localEvalModel)
      fd.append('csv_question_col', qCol)
      fd.append('csv_answer_col', aCol)
      fd.append('csv_embed_model', embedModel)
      fd.append('csv_collection_name', collectionName)
      fd.append('csv_smart_rag', String(smartRag))
      fd.append('csv_retrieval_mode', retrievalMode)
      fd.append('csv_score_threshold', String(scoreTh))
      fd.append('rag_mode', ragModeApi)
      fd.append('k', String(k))
      fd.append('thinking_enabled', String(think))
      fd.append('qa_models_json', JSON.stringify(qaSelected))
      if (openaiKey) fd.append('openai_api_key', openaiKey)
      const { data } = await api.post('/api/jobs/csv-pipeline', fd)
      return data.job_id as string
    },
    onSuccess: (id) => setJobId(id),
  })

  const result = jobQ.data?.result as
    | { rows: Record<string, unknown>[]; errors: string[]; csv_text: string }
    | undefined

  const cols: GridColDef[] = useMemo(() => {
    if (!result?.rows?.length) return []
    const keys = Object.keys(result.rows[0])
    return keys.map((field) => ({ field, headerName: field, flex: 1, minWidth: 120 }))
  }, [result])

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        CSV&apos;den soruları değerlendir
      </Typography>

      <Typography variant="subtitle2">Değerlendirilecek QA modelleri</Typography>
      {ollamaErr ? <Alert severity="error">{ollamaErr}</Alert> : null}
      <QaModelPicker
        allModels={allModels}
        filteredEmbeddingCount={ollamaQ.data?.filtered_embedding_count ?? 0}
        selected={qaSelected}
        onChange={setQaSelected}
        customModels={customModels}
        onCustomModelsChange={setCustomModels}
      />

      <Stack direction="row" spacing={2} sx={{ mb: 2, flexWrap: 'wrap', alignItems: 'center' }}>
        <FormControlLabel
          control={
            <Switch checked={evalEnabled} onChange={(_, v) => setEvalEnabled(v)} />
          }
          label="Değerlendir"
        />
        {evalEnabled && (
          <>
            <TextField
              select
              label="Değerlendirme motoru"
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
              <TextField
                size="small"
                label="OpenAI model"
                value={evalModelName}
                onChange={(e) => setEvalModelName(e.target.value)}
              />
            ) : (
              <TextField
                select
                label="Yerel eval modeli"
                value={localEvalModel}
                onChange={(e) => setLocalEvalModel(e.target.value)}
                slotProps={{ select: { native: true } }}
                size="small"
              >
                {(allModels.length ? allModels : ['—']).map((m: string) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </TextField>
            )}
          </>
        )}
      </Stack>

      <Button variant="outlined" component="label" sx={{ mr: 1 }}>
        CSV yükle
        <input
          type="file"
          hidden
          accept=".csv"
          onChange={(e) => setCsvFile(e.target.files?.[0] ?? null)}
        />
      </Button>
      {csvFile && <Typography variant="caption">{csvFile.name}</Typography>}

      <Stack direction="row" spacing={2} sx={{ mt: 2 }}>
        <TextField label="Soru sütunu" value={qCol} onChange={(e) => setQCol(e.target.value)} />
        <TextField label="Cevap sütunu" value={aCol} onChange={(e) => setACol(e.target.value)} />
      </Stack>

      <FormControlLabel
        control={<Checkbox checked={useSample} onChange={(_, v) => setUseSample(v)} />}
        label="Varsayılan örnek CSV (sample_rag_input.csv)"
      />

      <CollectionPicker
        embedModels={embedModels}
        embedLabel="Embedding modeli (indexleme ile aynı olmalı)"
        ragTypeLabel="RAG modu (indeksleme ile aynı olmalı)"
        ragHelp="BM25 anahtar kelime tabanlıdır. Smart mod parent/child koleksiyonlarını kullanır."
        horizontal
        classicLabel="Klasik koleksiyon"
        smartLabel="Smart koleksiyon"
        embedModel={embedModel}
        onEmbedModel={setEmbedModel}
        ragType={ragType}
        onRagType={setRagType}
        collectionName={collectionName}
        onCollectionName={setCollectionName}
      />

      <RadioGroup row value={ragModeUi} onChange={(_, v) => setRagModeUi(v as any)}>
        {RAG_MODE_UI.map((r) => (
          <FormControlLabel key={r} value={r} control={<Radio />} label={r} />
        ))}
      </RadioGroup>

      <Stack direction="row" spacing={3} sx={{ my: 2, alignItems: 'center' }}>
        {ragModeApi !== 'no_rag' && (
          <TextField
            type="number"
            label="k (chunk)"
            value={k}
            onChange={(e) => setK(+e.target.value)}
            size="small"
            sx={{ width: 120 }}
          />
        )}
        <FormControlLabel
          control={<Switch checked={think} onChange={(_, v) => setThink(v)} />}
          label="Thinking modu"
        />
      </Stack>

      {ragModeApi !== 'no_rag' && retrievalMode === 'vector' && (
        <Box sx={{ maxWidth: 480, mb: 2 }}>
          <Typography variant="caption">Minimum eşleşme skoru: {scoreTh}</Typography>
          <Slider
            min={0.1}
            max={1}
            step={0.05}
            value={scoreTh}
            onChange={(_, v) => setScoreTh(v as number)}
          />
        </Box>
      )}

      {evalBackend === 'OpenAI' && (
        <TextField
          fullWidth
          size="small"
          label="OpenAI API key (opsiyonel — boşsa ortam değişkeni)"
          value={openaiKey}
          onChange={(e) => setOpenaiKey(e.target.value)}
          sx={{ mb: 2 }}
        />
      )}

      <Button
        variant="contained"
        disabled={startMut.isPending}
        onClick={() => startMut.mutate()}
      >
        Pipeline&apos;ı çalıştır
      </Button>

      {jobId && jobQ.isFetching && jobQ.data?.status === 'running' && <LinearProgress sx={{ mt: 2 }} />}

      {jobQ.data?.status === 'failed' && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {jobQ.data.error}
        </Alert>
      )}

      {result?.errors?.map((e, i) => (
        <Alert key={i} severity="error" sx={{ mt: 1 }}>
          {e}
        </Alert>
      ))}

      {jobQ.data?.status === 'completed' && result && (
        <Box sx={{ mt: 2 }}>
          {!result.rows?.length ? (
            <Alert severity="warning">Hiç satır üretilmedi.</Alert>
          ) : (
            <>
              <Alert severity="success">
                Pipeline tamamlandı. Toplam {result.rows.length} satır.
              </Alert>
              <Box sx={{ height: 480, width: '100%', mt: 2 }}>
                <DataGrid rows={result.rows.map((r, id) => ({ id, ...r }))} columns={cols} />
              </Box>
              <Button
                variant="outlined"
                sx={{ mt: 1 }}
                component="a"
                href={URL.createObjectURL(
                  new Blob([result.csv_text], { type: 'text/csv;charset=utf-8' }),
                )}
                download="output.csv"
              >
                Sonuç CSV&apos;yi indir
              </Button>
            </>
          )}
        </Box>
      )}
    </Box>
  )
}
