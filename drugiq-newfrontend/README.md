# DrugIQ — Clean React + JavaScript + HTML/CSS UI

This version intentionally keeps the frontend minimal.

## Frontend stack
- React
- JavaScript (JSX)
- HTML
- CSS
- Native browser PDF opening for the real document endpoint

There is no TypeScript, Tailwind CSS, Lucide, Shadcn, Radix, React Router, or UI component library.

Vite is used only as the local React development/build tool.

## Run
```bash
npm install
npm run dev
```

## Production build
```bash
npm run build
```

## Backend integration
The UI integrates with the DrugIQ Flask API backend.
The source viewer opens a URL containing `file`, `page`, `drug`, and `snippets` query parameters.

