<template>
    <div :class="['message', message.sender]">
        <div class="message-content">
            {{ message.text }}
        </div>
    </div>
</template>

<script setup>
/**
 * MessageBubble - Zeigt eine einzelne Chat-Nachricht an
 * 
 * Props:
 * - message: { sender: 'user' | 'bot' | 'system', text: string }
 */
defineProps({
    message: {
        type: Object,
        required: true
    }
})
</script>

<style scoped>
.message {
    max-width: 85%;
    padding: 12px 16px;
    border-radius: 16px;
    line-height: 1.5;
    font-size: 0.95rem;
    animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* User Nachricht - rechts, lila */
.message.user {
    align-self: flex-end;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-bottom-right-radius: 4px;
}

/* Bot Nachricht - links, weiß */
.message.bot {
    align-self: flex-start;
    background: white;
    color: #333;
    border-bottom-left-radius: 4px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}

/* System Nachricht - zentriert, gelb */
.message.system {
    align-self: center;
    background: #fff3cd;
    color: #856404;
    font-size: 0.85rem;
    padding: 8px 16px;
    border-radius: 8px;
}
</style>
