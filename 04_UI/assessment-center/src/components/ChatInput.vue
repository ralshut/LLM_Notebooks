<template>
    <div class="input-area">
        <input
            v-model="inputText"
            @keyup.enter="handleSend"
            :placeholder="placeholder"
            :disabled="disabled"
            ref="inputRef"
        />
        <button 
            @click="handleSend" 
            :disabled="disabled || !inputText.trim()"
        >
            {{ disabled ? '...' : 'Senden' }}
        </button>
    </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'

/**
 * ChatInput - Eingabefeld mit Senden-Button
 * 
 * Props:
 * - disabled: boolean - Deaktiviert das Input während Loading
 * - placeholder: string - Platzhalter-Text
 * 
 * Emits:
 * - send: Wird ausgelöst wenn User sendet (mit Text als Payload)
 */
const props = defineProps({
    disabled: {
        type: Boolean,
        default: false
    },
    placeholder: {
        type: String,
        default: 'Deine Nachricht...'
    }
})

const emit = defineEmits(['send'])

const inputText = ref('')
const inputRef = ref(null)

function handleSend() {
    if (!inputText.value.trim() || props.disabled) return
    
    emit('send', inputText.value)
    inputText.value = ''
}

// Auto-Focus beim Mounten
onMounted(() => {
    inputRef.value?.focus()
})
</script>

<style scoped>
.input-area {
    padding: 20px;
    border-top: 1px solid #eee;
    display: flex;
    gap: 10px;
    background: white;
}

input {
    flex: 1;
    padding: 12px 16px;
    border: 2px solid #e9ecef;
    border-radius: 25px;
    outline: none;
    font-size: 0.95rem;
    transition: border-color 0.2s;
}

input:focus {
    border-color: #667eea;
}

input:disabled {
    background: #f8f9fa;
    cursor: not-allowed;
}

button {
    padding: 12px 24px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 25px;
    cursor: pointer;
    font-weight: 600;
    transition: transform 0.2s, opacity 0.2s;
}

button:hover:not(:disabled) {
    transform: scale(1.02);
}

button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
}
</style>
