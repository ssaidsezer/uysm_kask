import Box from '@mui/material/Box'
import FormControlLabel from '@mui/material/FormControlLabel'
import Stack from '@mui/material/Stack'
import Switch from '@mui/material/Switch'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import type { ModelProfile } from '../../pages/ModelProfilesPage'

export type EvalSavedProfilesSectionProps = {
  qaProfiles: ModelProfile[]
  evalProfiles: ModelProfile[]
  useSavedQaDefaults: boolean
  onUseSavedQaDefaults: (v: boolean) => void
  bulkQaProfileId: string
  onBulkQaProfileId: (id: string) => void
  thinkingEnabled: boolean
  onThinkingEnabled: (v: boolean) => void
  thinkingLabel: string
  useSavedEvalDefaults: boolean
  onUseSavedEvalDefaults: (v: boolean) => void
  evalProfileId: string
  onEvalProfileId: (id: string) => void
  /** `boxed`: bordered panel (chat page). `section`: title + stack only (CSV page). */
  layout?: 'boxed' | 'section'
  sectionTitle?: string
  stackMaxWidth?: number | string
}

export function EvalSavedProfilesSection({
  qaProfiles,
  evalProfiles,
  useSavedQaDefaults,
  onUseSavedQaDefaults,
  bulkQaProfileId,
  onBulkQaProfileId,
  thinkingEnabled,
  onThinkingEnabled,
  thinkingLabel,
  useSavedEvalDefaults,
  onUseSavedEvalDefaults,
  evalProfileId,
  onEvalProfileId,
  layout = 'boxed',
  sectionTitle = 'Kayıtlı model profilleri',
  stackMaxWidth,
}: EvalSavedProfilesSectionProps) {
  const inner = (
    <Stack spacing={1} sx={stackMaxWidth != null ? { maxWidth: stackMaxWidth } : undefined}>
      <FormControlLabel
        control={
          <Switch checked={useSavedQaDefaults} onChange={(_, v) => onUseSavedQaDefaults(v)} />
        }
        label="QA: modele göre varsayılan profili kullan"
      />
      <TextField
        select
        size="small"
        label="Tüm seçili QA modelleri için profil (opsiyonel)"
        value={bulkQaProfileId}
        onChange={(e) => onBulkQaProfileId(e.target.value)}
        slotProps={{ select: { native: true } }}
        fullWidth
      >
        <option value="">— Profil seçme —</option>
        {qaProfiles.map((p) => (
          <option key={p.id} value={p.id}>
            {p.name} ({p.model_name})
          </option>
        ))}
      </TextField>
      <FormControlLabel
        control={<Switch checked={thinkingEnabled} onChange={(_, v) => onThinkingEnabled(v)} />}
        label={thinkingLabel}
      />
      <FormControlLabel
        control={
          <Switch checked={useSavedEvalDefaults} onChange={(_, v) => onUseSavedEvalDefaults(v)} />
        }
        label="Eval: modele göre varsayılan profili kullan"
      />
      <TextField
        select
        size="small"
        label="Eval profili (opsiyonel)"
        value={evalProfileId}
        onChange={(e) => onEvalProfileId(e.target.value)}
        slotProps={{ select: { native: true } }}
        fullWidth
      >
        <option value="">— Profil seçme —</option>
        {evalProfiles.map((p) => (
          <option key={p.id} value={p.id}>
            {p.name} ({p.model_name})
          </option>
        ))}
      </TextField>
    </Stack>
  )

  if (layout === 'section') {
    return (
      <Box>
        <Typography variant="subtitle2" sx={{ mb: 1 }}>
          {sectionTitle}
        </Typography>
        {inner}
      </Box>
    )
  }

  return (
    <Box
      sx={{
        mb: 2,
        p: 2,
        border: '1px solid',
        borderColor: 'divider',
        borderRadius: 2,
      }}
    >
      <Typography variant="subtitle2" sx={{ mb: 1 }}>
        {sectionTitle}
      </Typography>
      {inner}
    </Box>
  )
}
