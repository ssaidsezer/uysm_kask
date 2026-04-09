import Box from '@mui/material/Box'
import FormControlLabel from '@mui/material/FormControlLabel'
import Stack from '@mui/material/Stack'
import Switch from '@mui/material/Switch'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import type { EvalBackendChoice } from '../../hooks/useEvalPageBasics'

export type EvalBackendSectionProps = {
  title?: string
  evalEnabled: boolean
  onEvalEnabled: (v: boolean) => void
  evalBackend: EvalBackendChoice
  onEvalBackend: (v: EvalBackendChoice) => void
  evalModelName: string
  onEvalModelName: (v: string) => void
  localEvalModel: string
  onLocalEvalModel: (v: string) => void
  openaiKey: string
  onOpenaiKey: (v: string) => void
  /** Options shown for local eval dropdown (parent may add placeholder entries). */
  localModelOptions: string[]
  variant: 'gridCard' | 'inlineRow'
  motorLabel?: string
  localEvalLabel?: string
  openaiKeyFieldSx?: object
}

export function EvalBackendSection({
  title = 'Değerlendirme',
  evalEnabled,
  onEvalEnabled,
  evalBackend,
  onEvalBackend,
  evalModelName,
  onEvalModelName,
  localEvalModel,
  onLocalEvalModel,
  openaiKey,
  onOpenaiKey,
  localModelOptions,
  variant,
  motorLabel = 'Motor',
  localEvalLabel = 'Yerel eval',
  openaiKeyFieldSx,
}: EvalBackendSectionProps) {
  const enabledFields = evalEnabled && (
    <Stack spacing={1} sx={{ mt: 1 }}>
      <TextField
        select
        label={motorLabel}
        value={evalBackend}
        onChange={(e) => onEvalBackend(e.target.value as EvalBackendChoice)}
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
            onChange={(e) => onEvalModelName(e.target.value)}
          />
          <TextField
            size="small"
            label="OpenAI API key (opsiyonel)"
            value={openaiKey}
            onChange={(e) => onOpenaiKey(e.target.value)}
            sx={openaiKeyFieldSx}
          />
        </>
      ) : (
        <TextField
          select
          size="small"
          label={localEvalLabel}
          value={localEvalModel}
          onChange={(e) => onLocalEvalModel(e.target.value)}
          slotProps={{ select: { native: true } }}
        >
          {localModelOptions.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </TextField>
      )}
    </Stack>
  )

  if (variant === 'gridCard') {
    return (
      <Box sx={{ p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
        <Typography variant="subtitle2" sx={{ mb: 1 }}>
          {title}
        </Typography>
        <FormControlLabel
          control={<Switch checked={evalEnabled} onChange={(_, v) => onEvalEnabled(v)} />}
          label="Değerlendir"
        />
        {enabledFields}
      </Box>
    )
  }

  return (
    <Stack direction="row" spacing={2} sx={{ flexWrap: 'wrap', alignItems: 'center' }}>
      <FormControlLabel
        control={<Switch checked={evalEnabled} onChange={(_, v) => onEvalEnabled(v)} />}
        label="Değerlendir"
      />
      {evalEnabled && (
        <>
          <TextField
            select
            label={motorLabel}
            value={evalBackend}
            onChange={(e) => onEvalBackend(e.target.value as EvalBackendChoice)}
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
                onChange={(e) => onEvalModelName(e.target.value)}
              />
              <TextField
                size="small"
                label="OpenAI API key (opsiyonel)"
                value={openaiKey}
                onChange={(e) => onOpenaiKey(e.target.value)}
                sx={openaiKeyFieldSx}
              />
            </>
          ) : (
            <TextField
              select
              label={localEvalLabel}
              value={localEvalModel}
              onChange={(e) => onLocalEvalModel(e.target.value)}
              slotProps={{ select: { native: true } }}
              size="small"
            >
              {localModelOptions.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </TextField>
          )}
        </>
      )}
    </Stack>
  )
}
