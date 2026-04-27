import { ref } from 'vue'
import { sendMessage, getScore } from '../services/api.js'

// Composable für das Assessment Center
export function useAssessment() {
    const chatHistory = ref([{ sender: 'bot', text: 'Willkommen! Ich bin Alex. Wie heißt du?' }])
    const loading = ref(false)
    const currentPhase = ref('host') 
    const finalResult = ref('')

    // Hauptfunktion für User-Interaktion
    async function handleUserMessage(text) {
        chatHistory.value.push({ sender: 'user', text })
        loading.value = true

        try {
            const res = await sendMessage(currentPhase.value, text)
            chatHistory.value.push({ sender: 'bot', text: res.text })

            if (res.nextPhase) {
                currentPhase.value = res.nextPhase

                if (res.nextPhase === 'angry_customer') {
                    setTimeout(() => chatHistory.value.push({ sender: 'system', text: '📞 Ein Anruf kommt rein...' }), 500)
                }
                
                if (res.nextPhase === 'scorer') {
                    await runScoring()
                }
            }
        } catch (error) {
            chatHistory.value.push({ sender: 'system', text: error.message })
        } finally {
            loading.value = false
        }
    }
    async function runScoring() {
        chatHistory.value.push({ sender: 'system', text: '⏳ Auswertung...' })
        try {
            finalResult.value = await getScore(chatHistory.value)
            currentPhase.value = 'result'
        } catch (e) {
            chatHistory.value.push({ sender: 'system', text: 'Fehler beim Scoring.' })
        }
    }

    function restart() {
        chatHistory.value = [{ sender: 'bot', text: 'Neustart. Wie heißt du?' }]
        currentPhase.value = 'host'
        finalResult.value = ''
    }

    return { chatHistory, loading, currentPhase, finalResult, handleUserMessage, restart }
}