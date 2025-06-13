package dev.langchain4j.model.chat;

import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Generic executor interface that defines a chat interaction
 */
public interface ChatExecutor {
    /**
     * Execute a chat request
     * @return The response
     */
    ChatResponse execute();

    /**
     * Executes a chat request using the provided chat memory.
     *
     * @param chatMemory The chat memory containing the context of the conversation.
     *                   It provides the history of messages required for proper interaction with the chat language model.
     * @return A response object containing the AI's response and additional metadata.
     * @see #execute(List)
     */
    default ChatResponse execute(ChatMemory chatMemory) {
        var messages = Optional.ofNullable(chatMemory).map(ChatMemory::messages).orElseGet(ArrayList::new);

        return execute(messages);
    }

    /**
     * Executes a chat request using the provided chat messages
     * @param chatMessages The chat messages containing the context of the conversation.
     *                     It provides the history of messages required for proper interaction with the chat model
     * @return A response object containing the AI's response and additional metadata.
     */
    ChatResponse execute(List<ChatMessage> chatMessages);

    /**
     * Creates a new {@link Builder} instance for constructing {@link ChatExecutor} objects
     * that perform synchronous chat requests.
     *
     * @return A new {@link Builder} instance to configure and build a {@link ChatExecutor}.
     */
    static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for constructing instances of {@link ChatExecutor}.
     *
     * This builder provides a fluent API for setting required components
     * like {@link ChatRequest}, and for building an instance of the {@link ChatExecutor}.
     */
    class Builder {
        protected ChatRequest chatRequest;
        protected ChatModel chatModel;

        protected Builder() {}

        /**
         * Sets the {@link ChatRequest} instance for the synchronousBuilder.
         * The {@link ChatRequest} encapsulates the input messages and parameters required
         * to generate a response from the chat model.
         *
         * @param chatRequest the {@link ChatRequest} containing the input messages and parameters
         * @return the updated Builder instance
         */
        public Builder chatRequest(ChatRequest chatRequest) {
            this.chatRequest = chatRequest;
            return this;
        }

        /**
         * Sets the {@link ChatModel} instance for the Builder.
         * The {@link ChatModel} represents a language model that provides a chat API.
         *
         * @param chatModel the {@link ChatModel} to be used by the Builder
         * @return the updated Builder instance
         */
        public Builder chatModel(ChatModel chatModel) {
            this.chatModel = chatModel;
            return this;
        }

        /**
         * Constructs and returns an instance of {@link ChatExecutor}.
         * Ensures that all required parameters have been appropriately set
         * before building the {@link ChatExecutor}.
         *
         * @return a fully constructed {@link ChatExecutor} instance
         */
        public ChatExecutor build() {
            return new DefaultChatExecutor(this);
        }
    }

    /**
     * Exception thrown when an attempt is made to execute a chat request without a valid chat model.
     * This typically occurs within the {@code ChatExecutor} when no {@code ChatModel} is provided or configured.
     * <p>
     *     This is intended for output guardrails, but can be caught in other places when streaming needs to be converted
     *     to synchronous.
     * </p>
     */
    class NoChatModelFoundException extends RuntimeException {
        public NoChatModelFoundException(String message) {
            super(message);
        }

        public NoChatModelFoundException() {
            this("Can not invoke ChatExecutor without a ChatModel");
        }
    }
}
