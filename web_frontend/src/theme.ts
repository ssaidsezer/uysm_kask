import { createTheme } from '@mui/material/styles'

export const appTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: '#4a90d9' },
    background: { default: '#0e1117', paper: '#1e1e1e' },
  },
  typography: {
    fontFamily: '"Source Sans Pro", "Segoe UI", sans-serif',
  },
  components: {
    MuiTab: { styleOverrides: { root: { textTransform: 'none' } } },
    MuiButton: { styleOverrides: { root: { textTransform: 'none' } } },
  },
})
