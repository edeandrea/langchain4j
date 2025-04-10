package dev.langchain4j.service;

import static dev.langchain4j.internal.Utils.copyIfNotNull;
import static dev.langchain4j.internal.ValidationUtils.ensureNotEmpty;
import static dev.langchain4j.internal.ValidationUtils.ensureNotNull;

import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.guardrail.GuardrailParams.CommonGuardrailParams;
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.service.tool.ToolExecutor;
import java.util.List;
import java.util.Map;
import org.jspecify.annotations.Nullable;

/**
 * Parameters for creating an {@link AiServiceTokenStream}.
 */
public class AiServiceTokenStreamParameters {
    private final List<ChatMessage> messages;
    private final List<ToolSpecification> toolSpecifications;
    private final Map<String, ToolExecutor> toolExecutors;
    private final List<Content> retrievedContents;
    private final AiServiceContext context;
    private final Object memoryId;

    @Nullable
    private final CommonGuardrailParams commonGuardrailParams;

    @Nullable
    private final Object methodKey;

    protected AiServiceTokenStreamParameters(Builder builder) {
        this.messages = ensureNotEmpty(builder.messages, "messages");
        this.toolSpecifications = copyIfNotNull(builder.toolSpecifications);
        this.toolExecutors = copyIfNotNull(builder.toolExecutors);
        this.retrievedContents = builder.retrievedContents;
        this.context = ensureNotNull(builder.context, "context");
        this.memoryId = ensureNotNull(builder.memoryId, "memoryId");
        this.commonGuardrailParams = builder.commonGuardrailParams;
        this.methodKey = builder.methodKey;
        ensureNotNull(context.streamingChatModel, "streamingChatModel");
    }

    /**
     * @return the messages
     */
    public List<ChatMessage> messages() {
        return messages;
    }

    /**
     * @return the tool specifications
     */
    public List<ToolSpecification> toolSpecifications() {
        return toolSpecifications;
    }

    /**
     * @return the tool executors
     */
    public Map<String, ToolExecutor> toolExecutors() {
        return toolExecutors;
    }

    /**
     * @return the retrieved contents
     */
    public List<Content> gretrievedContents() {
        return retrievedContents;
    }

    /**
     * @return the AI service context
     */
    public AiServiceContext context() {
        return context;
    }

    /**
     * @return the memory ID
     */
    public Object memoryId() {
        return memoryId;
    }

    /**
     * Retrieves the common parameters shared across guardrail checks for validating interactions
     * between a user and a language model, if available.
     *
     * @return the {@link CommonGuardrailParams} containing chat memory, user message template,
     * and additional variables required for guardrail processing, or null if not set.
     */
    @Nullable
    public CommonGuardrailParams commonGuardrailParams() {
        return commonGuardrailParams;
    }

    /**
     * Retrieves the method key associated with this instance.
     *
     * @return the method key as an Object
     */
    @Nullable
    public Object methodKey() {
        return methodKey;
    }

    /**
     * Creates a new builder for {@link AiServiceTokenStreamParameters}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for {@link AiServiceTokenStreamParameters}.
     */
    public static class Builder {
        private List<ChatMessage> messages;
        private List<ToolSpecification> toolSpecifications;
        private Map<String, ToolExecutor> toolExecutors;
        private List<Content> retrievedContents;
        private AiServiceContext context;
        private Object memoryId;
        private CommonGuardrailParams commonGuardrailParams;
        private Object methodKey;

        protected Builder() {}

        /**
         * Sets the messages.
         *
         * @param messages the messages
         * @return this builder
         */
        public Builder messages(List<ChatMessage> messages) {
            this.messages = messages;
            return this;
        }

        /**
         * Sets the tool specifications.
         *
         * @param toolSpecifications the tool specifications
         * @return this builder
         */
        public Builder toolSpecifications(List<ToolSpecification> toolSpecifications) {
            this.toolSpecifications = toolSpecifications;
            return this;
        }

        /**
         * Sets the tool executors.
         *
         * @param toolExecutors the tool executors
         * @return this builder
         */
        public Builder toolExecutors(Map<String, ToolExecutor> toolExecutors) {
            this.toolExecutors = toolExecutors;
            return this;
        }

        /**
         * Sets the retrieved contents.
         *
         * @param retrievedContents the retrieved contents
         * @return this builder
         */
        public Builder retrievedContents(List<Content> retrievedContents) {
            this.retrievedContents = retrievedContents;
            return this;
        }

        /**
         * Sets the AI service context.
         *
         * @param context the AI service context
         * @return this builder
         */
        public Builder context(AiServiceContext context) {
            this.context = context;
            return this;
        }

        /**
         * Sets the memory ID.
         *
         * @param memoryId the memory ID
         * @return this builder
         */
        public Builder memoryId(Object memoryId) {
            this.memoryId = memoryId;
            return this;
        }

        /**
         * Sets the common guardrail parameters for validating interactions between a user and a language model.
         *
         * @param commonGuardrailParams an instance of {@link CommonGuardrailParams} containing the shared parameters
         *                              required for guardrail checks, such as chat memory, user message template,
         *                              and additional variables.
         * @return this builder instance.
         */
        public Builder commonGuardrailParams(CommonGuardrailParams commonGuardrailParams) {
            this.commonGuardrailParams = commonGuardrailParams;
            return this;
        }

        /**
         * Sets the method key.
         *
         * @param methodKey the method key
         * @return this builder
         */
        public Builder methodKey(Object methodKey) {
            this.methodKey = methodKey;
            return this;
        }

        /**
         * Builds a new {@link AiServiceTokenStreamParameters}.
         *
         * @return a new {@link AiServiceTokenStreamParameters}
         */
        public AiServiceTokenStreamParameters build() {
            return new AiServiceTokenStreamParameters(this);
        }
    }
}
