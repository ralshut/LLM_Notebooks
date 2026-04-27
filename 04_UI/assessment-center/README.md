# AI Assessment Center - Frontend

Ein Vue.js Frontend für das KI-gestützte Assessment Center mit n8n Backend.

## Schnellstart

```bash
# 1. Dependencies installieren
npm install

# 2. Entwicklungsserver starten
npm run dev
```

Danach öffne http://localhost:5173 im Browser.

## Projektstruktur

```
src/
├── components/
│   ├── ChatWindow.vue      # Der Chat-Bereich mit Nachrichten
│   ├── MessageBubble.vue   # Einzelne Chat-Nachricht (User/Bot/System)
│   ├── ChatInput.vue       # Eingabefeld + Senden-Button
│   └── ResultCard.vue      # Ergebnis-Anzeige nach dem Scoring
├── composables/
│   └── useAssessment.js    # Logik und State
├── services/
│   └── api.js              # Hier Webhooks eintragen
├── App.vue                 # Orchestrator
├── main.js                 # Vue-App Initialisierung
└── style.css               # Globale Styles
```

## Aufgaben

### Aufgabe 1: Webhook URLs eintragen
Öffne `src/services/api.js` und ersetze die Platzhalter mit deinen n8n Webhook URLs:

```javascript
const URLS = {
    host:   'http://localhost:5678/webhook-test/DEINE-HOST-ID',
    angry:  'http://localhost:5678/webhook-test/DEINE-ANGRY-ID',
    scorer: 'http://localhost:5678/webhook-test/DEINE-SCORER-ID'
}
```

### Aufgabe 2: API-Service verstehen
Schau dir die Funktionen in `api.js` an:
- `sendMessage(phase, text)` - Sendet eine Nachricht an den entsprechenden Workflow
- `getScore(chatHistory)` - Sendet den kompletten Verlauf an den Scorer

### Aufgabe 3: Komponenten erkunden
1. Öffne `App.vue` - hier ist der State (chatHistory, currentPhase)
2. Öffne `ChatWindow.vue` - hier werden die Nachrichten gerendert
3. Öffne `MessageBubble.vue` - hier wird eine einzelne Nachricht dargestellt

## 🔧 CORS Problem lösen

Falls du einen CORS-Fehler siehst:
1. Installiere die Chrome-Extension "Allow CORS: Access-Control-Allow-Origin"
2. Aktiviere sie (Icon wird bunt)
3. Lade die Seite neu

## 📚 Vue.js Kurzreferenz

### Reaktive Daten (State)
```javascript
const chatHistory = ref([])  // ref() macht Variablen reaktiv
chatHistory.value.push(...)  // Zugriff über .value
```

### Template Syntax
```html
<div v-for="msg in chatHistory">  <!-- Loop -->
<div v-if="loading">              <!-- Bedingung -->
<button @click="sendMessage">     <!-- Event Handler -->
<input v-model="userInput">       <!-- Two-Way Binding -->
```

### Komponenten
```html
<MessageBubble :message="msg" />  <!-- Props übergeben -->
```

## 🐛 Troubleshooting

| Problem | Lösung |
|---------|--------|
| `npm install` schlägt fehl | Node.js Version prüfen (mind. v18) |
| CORS Error | Browser-Extension aktivieren |
| 404 Not registered | n8n Workflow auf "Listen" stellen |
| Keine Antwort | Prüfe ob Ollama läuft (`ollama serve`) |
