package dev.langchain4j.model.azure;

import static dev.langchain4j.model.output.FinishReason.LENGTH;
import static dev.langchain4j.model.output.FinishReason.STOP;
import static java.util.concurrent.TimeUnit.SECONDS;
import static org.assertj.core.api.Assertions.assertThat;

import dev.langchain4j.model.StreamingResponseHandler;
import dev.langchain4j.model.language.StreamingLanguageModel;
import dev.langchain4j.model.output.Response;
import java.util.concurrent.CompletableFuture;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

@EnabledIfEnvironmentVariable(named = "AZURE_OPENAI_KEY", matches = ".+")
class AzureOpenAiStreamingLanguageModelIT {

    StreamingLanguageModel model = AzureModelBuilders.streamingLanguageModelBuilder()
            .deploymentName("gpt-35-turbo-instruct-0914")
            .temperature(0.0)
            .maxTokens(10)
            .logRequestsAndResponses(true)
            .build();

    @Test
    void should_stream_answer_and_finish_reason_stop() throws Exception {

        CompletableFuture<String> futureAnswer = new CompletableFuture<>();
        CompletableFuture<Response<String>> futureResponse = new CompletableFuture<>();

        model.generate("The capital of France is: ", new StreamingResponseHandler<>() {

            private final StringBuilder answerBuilder = new StringBuilder();

            @Override
            public void onNext(String token) {
                answerBuilder.append(token);
            }

            @Override
            public void onComplete(Response<String> response) {
                futureAnswer.complete(answerBuilder.toString());
                futureResponse.complete(response);
            }

            @Override
            public void onError(Throwable error) {
                futureAnswer.completeExceptionally(error);
                futureResponse.completeExceptionally(error);
            }
        });

        String answer = futureAnswer.get(30, SECONDS);
        Response<String> response = futureResponse.get(30, SECONDS);

        assertThat(answer).containsIgnoringCase("Paris");
        assertThat(response.content()).isEqualTo(answer);

        assertThat(response.tokenUsage()).isNotNull();
        assertThat(response.tokenUsage().inputTokenCount()).isGreaterThan(0);
        assertThat(response.tokenUsage().outputTokenCount()).isGreaterThan(0);
        assertThat(response.tokenUsage().totalTokenCount()).isEqualTo(
                response.tokenUsage().inputTokenCount()
                        + response.tokenUsage().outputTokenCount());

        assertThat(response.finishReason()).isEqualTo(STOP);
    }

    @Test
    void should_stream_answer_and_finish_reason_length() throws Exception {

        CompletableFuture<Response<String>> futureResponse = new CompletableFuture<>();

        model.generate("Describe the capital of France in 50 words: ", new StreamingResponseHandler<>() {

            @Override
            public void onNext(String token) {}

            @Override
            public void onComplete(Response<String> response) {
                futureResponse.complete(response);
            }

            @Override
            public void onError(Throwable error) {
                futureResponse.completeExceptionally(error);
            }
        });

        Response<String> response = futureResponse.get(30, SECONDS);

        assertThat(response.finishReason()).isEqualTo(LENGTH);
    }
}
