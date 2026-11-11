/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: {
          DEFAULT: "#F5F3EC",
          subtle: "#F9F8F4",
          muted: "#EBE7DC",
          card: "#FFFFFF",
        },
        sidebar: {
          DEFAULT: "#131316",
          dark: "#0E0E11",
          hover: "#1D1D23",
          active: "#25252D",
          border: "#202028",
        },
        pastel: {
          yellow: "#FEF3C7",
          yellowDark: "#D97706",
          pink: "#FCE7F3",
          pinkDark: "#DB2777",
          pinkAccent: "#F472B6",
          green: "#DCFCE7",
          greenDark: "#15803D",
          blue: "#DBEAFE",
          blueDark: "#2563EB",
          purple: "#EDE9FE",
        },
        soc: {
          normal: "#16A34A",
          warning: "#D97706",
          critical: "#DC2626",
          border: "#E2DED4",
          surface: "#FFFFFF",
        },
      },
      fontFamily: {
        sans: [
          "Inter",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        mono: [
          "JetBrains Mono",
          "Fira Code",
          "SF Mono",
          "monospace",
        ],
      },
      borderRadius: {
        '2xl': '1rem',
        '3xl': '1.5rem',
        '4xl': '2rem',
      },
      boxShadow: {
        'subtle': '0 1px 3px rgba(0, 0, 0, 0.05), 0 1px 2px rgba(0, 0, 0, 0.03)',
        'card': '0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -1px rgba(0, 0, 0, 0.02)',
      },
    },
  },
  plugins: [],
}
