<template>
    <div class="chat-window" ref="chatContainer">
        <!-- Alle Nachrichten rendern -->
        <MessageBubble 
            v-for="(msg, index) in messages" 
            :key="index" 
            :message="msg" 
        />
        
        <!-- Typing Indicator -->
        <div v-if="loading" class="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
        </div>
    </div>
</template>

<script setup>
import { ref, watch, nextTick } from 'vue'
import MessageBubble from './MessageBubble.vue'

/**
 * ChatWindow - Container für alle Chat-Nachrichten
 * 
 * Props:
 * - messages: Array von { sender, text } Objekten
 * - loading: boolean - Zeigt Typing-Indicator
 */
const props = defineProps({
    messages: {
        type: Array,
        required: true
    },
    loading: {
        type: Boolean,
        default: false
    }
})

const chatContainer = ref(null)

// Auto-Scroll wenn neue Nachrichten kommen
watch(
    () => props.messages.length,
    async () => {
        await nextTick()
        scrollToBottom()
    }
)

watch(
    () => props.loading,
    async () => {
        await nextTick()
        scrollToBottom()
    }
)

function scrollToBottom() {
    if (chatContainer.value) {
        chatContainer.value.scrollTop = chatContainer.value.scrollHeight
    }
}
</script>

<style scoped>
.chat-window {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    background: #f8f9fa;
}

/* Typing Indicator Animation */
.typing-indicator {
    align-self: flex-start;
    background: white;
    padding: 12px 16px;
    border-radius: 16px;
    border-bottom-left-radius: 4px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    display: flex;
    gap: 4px;
}

.typing-indicator span {
    width: 8px;
    height: 8px;
    background: #667eea;
    border-radius: 50%;
    animation: bounce 1.4s infinite ease-in-out both;
}

.typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
.typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
.typing-indicator span:nth-child(3) { animation-delay: 0s; }

@keyframes bounce {
    0%, 80%, 100% { transform: scale(0); }
    40% { transform: scale(1); }
}
</style>
