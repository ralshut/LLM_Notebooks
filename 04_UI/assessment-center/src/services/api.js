
// API Service für die Kommunikation mit n8n Webhooks


// ===========================================
// TODO: Ersetze diese URLs mit deinen n8n Webhook URLs, achte auf den genauen Pfad und darauf ob du den Webhook mit /webhook/ oder /webhook-test/ erstellt hast.
// ===========================================

const URLS = {
    host:   'http://localhost:5678/webhook-test/DEINE-HOST-ID', // Aufgabe 5.1
    angry:  'http://localhost:5678/webhook-test/DEINE-ANGRY-ID',
    scorer: 'http://localhost:5678/webhook-test/DEINE-SCORER-ID'
}


//Sendet eine Nachricht an den entsprechenden n8n Workflow
export async function sendMessage(phase, text) {
    const url = URLS[phase]
    if (!url || url.includes('DEINE-')) throw new Error('Webhook URL missing in api.js')

    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chatInput: text })
    })

    if (!response.ok) throw new Error(`HTTP Error: ${response.status}`)

    const data = await response.json()
    
    return {
        text: data.output || data.text || data.message || JSON.stringify(data),
        triggerSwitch: data.triggerSwitch === true,
        nextPhase: data.nextPhase
    }
}

// Sendet den kompletten Chatverlauf an den Scorer-Workflow
export async function getScore(chatHistory) {
    const url = URLS.scorer
    if (!url || url.includes('DEINE-')) throw new Error('Scorer URL missing')

    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fullHistory: chatHistory })
    })

    if (!response.ok) throw new Error(`HTTP Error: ${response.status}`)
    const data = await response.json()
    
    return data.output || data.text || data.feedback || JSON.stringify(data)
}