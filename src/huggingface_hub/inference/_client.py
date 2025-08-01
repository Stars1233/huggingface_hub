# coding=utf-8
# Copyright 2023-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Related resources:
#    https://huggingface.co/tasks
#    https://huggingface.co/docs/huggingface.js/inference/README
#    https://github.com/huggingface/huggingface.js/tree/main/packages/inference/src
#    https://github.com/huggingface/text-generation-inference/tree/main/clients/python
#    https://github.com/huggingface/text-generation-inference/blob/main/clients/python/text_generation/client.py
#    https://huggingface.slack.com/archives/C03E4DQ9LAJ/p1680169099087869
#    https://github.com/huggingface/unity-api#tasks
#
# Some TODO:
# - add all tasks
#
# NOTE: the philosophy of this client is "let's make it as easy as possible to use it, even if less optimized". Some
# examples of how it translates:
# - Timeout / Server unavailable is handled by the client in a single "timeout" parameter.
# - Files can be provided as bytes, file paths, or URLs and the client will try to "guess" the type.
# - Images are parsed as PIL.Image for easier manipulation.
# - Provides a "recommended model" for each task => suboptimal but user-wise quicker to get a first script running.
# - Only the main parameters are publicly exposed. Power users can always read the docs for more options.
import base64
import logging
import re
import warnings
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Literal, Optional, Union, overload

from requests import HTTPError

from huggingface_hub import constants
from huggingface_hub.errors import BadRequestError, InferenceTimeoutError
from huggingface_hub.inference._common import (
    TASKS_EXPECTING_IMAGES,
    ContentT,
    RequestParameters,
    _b64_encode,
    _b64_to_image,
    _bytes_to_dict,
    _bytes_to_image,
    _bytes_to_list,
    _get_unsupported_text_generation_kwargs,
    _import_numpy,
    _open_as_binary,
    _set_unsupported_text_generation_kwargs,
    _stream_chat_completion_response,
    _stream_text_generation_response,
    raise_text_generation_error,
)
from huggingface_hub.inference._generated.types import (
    AudioClassificationOutputElement,
    AudioClassificationOutputTransform,
    AudioToAudioOutputElement,
    AutomaticSpeechRecognitionOutput,
    ChatCompletionInputGrammarType,
    ChatCompletionInputMessage,
    ChatCompletionInputStreamOptions,
    ChatCompletionInputTool,
    ChatCompletionInputToolChoiceClass,
    ChatCompletionInputToolChoiceEnum,
    ChatCompletionOutput,
    ChatCompletionStreamOutput,
    DocumentQuestionAnsweringOutputElement,
    FillMaskOutputElement,
    ImageClassificationOutputElement,
    ImageClassificationOutputTransform,
    ImageSegmentationOutputElement,
    ImageSegmentationSubtask,
    ImageToImageTargetSize,
    ImageToTextOutput,
    ObjectDetectionOutputElement,
    Padding,
    QuestionAnsweringOutputElement,
    SummarizationOutput,
    SummarizationTruncationStrategy,
    TableQuestionAnsweringOutputElement,
    TextClassificationOutputElement,
    TextClassificationOutputTransform,
    TextGenerationInputGrammarType,
    TextGenerationOutput,
    TextGenerationStreamOutput,
    TextToSpeechEarlyStoppingEnum,
    TokenClassificationAggregationStrategy,
    TokenClassificationOutputElement,
    TranslationOutput,
    TranslationTruncationStrategy,
    VisualQuestionAnsweringOutputElement,
    ZeroShotClassificationOutputElement,
    ZeroShotImageClassificationOutputElement,
)
from huggingface_hub.inference._providers import PROVIDER_OR_POLICY_T, get_provider_helper
from huggingface_hub.utils import build_hf_headers, get_session, hf_raise_for_status
from huggingface_hub.utils._auth import get_token


if TYPE_CHECKING:
    import numpy as np
    from PIL.Image import Image

logger = logging.getLogger(__name__)


MODEL_KWARGS_NOT_USED_REGEX = re.compile(r"The following `model_kwargs` are not used by the model: \[(.*?)\]")


class InferenceClient:
    """
    Initialize a new Inference Client.

    [`InferenceClient`] aims to provide a unified experience to perform inference. The client can be used
    seamlessly with either the (free) Inference API, self-hosted Inference Endpoints, or third-party Inference Providers.

    Args:
        model (`str`, `optional`):
            The model to run inference with. Can be a model id hosted on the Hugging Face Hub, e.g. `meta-llama/Meta-Llama-3-8B-Instruct`
            or a URL to a deployed Inference Endpoint. Defaults to None, in which case a recommended model is
            automatically selected for the task.
            Note: for better compatibility with OpenAI's client, `model` has been aliased as `base_url`. Those 2
            arguments are mutually exclusive. If a URL is passed as `model` or `base_url` for chat completion, the `(/v1)/chat/completions` suffix path will be appended to the URL.
        provider (`str`, *optional*):
            Name of the provider to use for inference. Can be `"black-forest-labs"`, `"cerebras"`, `"cohere"`, `"fal-ai"`, `"featherless-ai"`, `"fireworks-ai"`, `"groq"`, `"hf-inference"`, `"hyperbolic"`, `"nebius"`, `"novita"`, `"nscale"`, `"openai"`, `"replicate"`, "sambanova"` or `"together"`.
            Defaults to "auto" i.e. the first of the providers available for the model, sorted by the user's order in https://hf.co/settings/inference-providers.
            If model is a URL or `base_url` is passed, then `provider` is not used.
        token (`str`, *optional*):
            Hugging Face token. Will default to the locally saved token if not provided.
            Note: for better compatibility with OpenAI's client, `token` has been aliased as `api_key`. Those 2
            arguments are mutually exclusive and have the exact same behavior.
        timeout (`float`, `optional`):
            The maximum number of seconds to wait for a response from the server. Defaults to None, meaning it will loop until the server is available.
        headers (`Dict[str, str]`, `optional`):
            Additional headers to send to the server. By default only the authorization and user-agent headers are sent.
            Values in this dictionary will override the default values.
        bill_to (`str`, `optional`):
            The billing account to use for the requests. By default the requests are billed on the user's account.
            Requests can only be billed to an organization the user is a member of, and which has subscribed to Enterprise Hub.
        cookies (`Dict[str, str]`, `optional`):
            Additional cookies to send to the server.
        proxies (`Any`, `optional`):
            Proxies to use for the request.
        base_url (`str`, `optional`):
            Base URL to run inference. This is a duplicated argument from `model` to make [`InferenceClient`]
            follow the same pattern as `openai.OpenAI` client. Cannot be used if `model` is set. Defaults to None.
        api_key (`str`, `optional`):
            Token to use for authentication. This is a duplicated argument from `token` to make [`InferenceClient`]
            follow the same pattern as `openai.OpenAI` client. Cannot be used if `token` is set. Defaults to None.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        provider: Optional[PROVIDER_OR_POLICY_T] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = None,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        proxies: Optional[Any] = None,
        bill_to: Optional[str] = None,
        # OpenAI compatibility
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        if model is not None and base_url is not None:
            raise ValueError(
                "Received both `model` and `base_url` arguments. Please provide only one of them."
                " `base_url` is an alias for `model` to make the API compatible with OpenAI's client."
                " If using `base_url` for chat completion, the `/chat/completions` suffix path will be appended to the base url."
                " When passing a URL as `model`, the client will not append any suffix path to it."
            )
        if token is not None and api_key is not None:
            raise ValueError(
                "Received both `token` and `api_key` arguments. Please provide only one of them."
                " `api_key` is an alias for `token` to make the API compatible with OpenAI's client."
                " It has the exact same behavior as `token`."
            )
        token = token if token is not None else api_key
        if isinstance(token, bool):
            # Legacy behavior: previously is was possible to pass `token=False` to disable authentication. This is not
            # supported anymore as authentication is required. Better to explicitly raise here rather than risking
            # sending the locally saved token without the user knowing about it.
            if token is False:
                raise ValueError(
                    "Cannot use `token=False` to disable authentication as authentication is required to run Inference."
                )
            warnings.warn(
                "Using `token=True` to automatically use the locally saved token is deprecated and will be removed in a future release. "
                "Please use `token=None` instead (default).",
                DeprecationWarning,
            )
            token = get_token()

        self.model: Optional[str] = base_url or model
        self.token: Optional[str] = token

        self.headers = {**headers} if headers is not None else {}
        if bill_to is not None:
            if (
                constants.HUGGINGFACE_HEADER_X_BILL_TO in self.headers
                and self.headers[constants.HUGGINGFACE_HEADER_X_BILL_TO] != bill_to
            ):
                warnings.warn(
                    f"Overriding existing '{self.headers[constants.HUGGINGFACE_HEADER_X_BILL_TO]}' value in headers with '{bill_to}'.",
                    UserWarning,
                )
            self.headers[constants.HUGGINGFACE_HEADER_X_BILL_TO] = bill_to

            if token is not None and not token.startswith("hf_"):
                warnings.warn(
                    "You've provided an external provider's API key, so requests will be billed directly by the provider. "
                    "The `bill_to` parameter is only applicable for Hugging Face billing and will be ignored.",
                    UserWarning,
                )

        # Configure provider
        self.provider = provider

        self.cookies = cookies
        self.timeout = timeout
        self.proxies = proxies

    def __repr__(self):
        return f"<InferenceClient(model='{self.model if self.model else ''}', timeout={self.timeout})>"

    @overload
    def _inner_post(  # type: ignore[misc]
        self, request_parameters: RequestParameters, *, stream: Literal[False] = ...
    ) -> bytes: ...

    @overload
    def _inner_post(  # type: ignore[misc]
        self, request_parameters: RequestParameters, *, stream: Literal[True] = ...
    ) -> Iterable[bytes]: ...

    @overload
    def _inner_post(
        self, request_parameters: RequestParameters, *, stream: bool = False
    ) -> Union[bytes, Iterable[bytes]]: ...

    def _inner_post(
        self, request_parameters: RequestParameters, *, stream: bool = False
    ) -> Union[bytes, Iterable[bytes]]:
        """Make a request to the inference server."""
        # TODO: this should be handled in provider helpers directly
        if request_parameters.task in TASKS_EXPECTING_IMAGES and "Accept" not in request_parameters.headers:
            request_parameters.headers["Accept"] = "image/png"

        with _open_as_binary(request_parameters.data) as data_as_binary:
            try:
                response = get_session().post(
                    request_parameters.url,
                    json=request_parameters.json,
                    data=data_as_binary,
                    headers=request_parameters.headers,
                    cookies=self.cookies,
                    timeout=self.timeout,
                    stream=stream,
                    proxies=self.proxies,
                )
            except TimeoutError as error:
                # Convert any `TimeoutError` to a `InferenceTimeoutError`
                raise InferenceTimeoutError(f"Inference call timed out: {request_parameters.url}") from error  # type: ignore

        try:
            hf_raise_for_status(response)
            return response.iter_lines() if stream else response.content
        except HTTPError as error:
            if error.response.status_code == 422 and request_parameters.task != "unknown":
                msg = str(error.args[0])
                if len(error.response.text) > 0:
                    msg += f"\n{error.response.text}\n"
                error.args = (msg,) + error.args[1:]
            raise

    def audio_classification(
        self,
        audio: ContentT,
        *,
        model: Optional[str] = None,
        top_k: Optional[int] = None,
        function_to_apply: Optional["AudioClassificationOutputTransform"] = None,
    ) -> List[AudioClassificationOutputElement]:
        """
        Perform audio classification on the provided audio content.

        Args:
            audio (Union[str, Path, bytes, BinaryIO]):
                The audio content to classify. It can be raw audio bytes, a local audio file, or a URL pointing to an
                audio file.
            model (`str`, *optional*):
                The model to use for audio classification. Can be a model ID hosted on the Hugging Face Hub
                or a URL to a deployed Inference Endpoint. If not provided, the default recommended model for
                audio classification will be used.
            top_k (`int`, *optional*):
                When specified, limits the output to the top K most probable classes.
            function_to_apply (`"AudioClassificationOutputTransform"`, *optional*):
                The function to apply to the model outputs in order to retrieve the scores.

        Returns:
            `List[AudioClassificationOutputElement]`: List of [`AudioClassificationOutputElement`] items containing the predicted labels and their confidence.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.audio_classification("audio.flac")
        [
            AudioClassificationOutputElement(score=0.4976358711719513, label='hap'),
            AudioClassificationOutputElement(score=0.3677836060523987, label='neu'),
            ...
        ]
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="audio-classification", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=audio,
            parameters={"function_to_apply": function_to_apply, "top_k": top_k},
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        return AudioClassificationOutputElement.parse_obj_as_list(response)

    def audio_to_audio(
        self,
        audio: ContentT,
        *,
        model: Optional[str] = None,
    ) -> List[AudioToAudioOutputElement]:
        """
        Performs multiple tasks related to audio-to-audio depending on the model (eg: speech enhancement, source separation).

        Args:
            audio (Union[str, Path, bytes, BinaryIO]):
                The audio content for the model. It can be raw audio bytes, a local audio file, or a URL pointing to an
                audio file.
            model (`str`, *optional*):
                The model can be any model which takes an audio file and returns another audio file. Can be a model ID hosted on the Hugging Face Hub
                or a URL to a deployed Inference Endpoint. If not provided, the default recommended model for
                audio_to_audio will be used.

        Returns:
            `List[AudioToAudioOutputElement]`: A list of [`AudioToAudioOutputElement`] items containing audios label, content-type, and audio content in blob.

        Raises:
            `InferenceTimeoutError`:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> audio_output = client.audio_to_audio("audio.flac")
        >>> for i, item in enumerate(audio_output):
        >>>     with open(f"output_{i}.flac", "wb") as f:
                    f.write(item.blob)
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="audio-to-audio", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=audio,
            parameters={},
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        audio_output = AudioToAudioOutputElement.parse_obj_as_list(response)
        for item in audio_output:
            item.blob = base64.b64decode(item.blob)
        return audio_output

    def automatic_speech_recognition(
        self,
        audio: ContentT,
        *,
        model: Optional[str] = None,
        extra_body: Optional[Dict] = None,
    ) -> AutomaticSpeechRecognitionOutput:
        """
        Perform automatic speech recognition (ASR or audio-to-text) on the given audio content.

        Args:
            audio (Union[str, Path, bytes, BinaryIO]):
                The content to transcribe. It can be raw audio bytes, local audio file, or a URL to an audio file.
            model (`str`, *optional*):
                The model to use for ASR. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. If not provided, the default recommended model for ASR will be used.
            extra_body (`Dict`, *optional*):
                Additional provider-specific parameters to pass to the model. Refer to the provider's documentation
                for supported parameters.
        Returns:
            [`AutomaticSpeechRecognitionOutput`]: An item containing the transcribed text and optionally the timestamp chunks.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.automatic_speech_recognition("hello_world.flac").text
        "hello world"
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="automatic-speech-recognition", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=audio,
            parameters={**(extra_body or {})},
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        return AutomaticSpeechRecognitionOutput.parse_obj_as_instance(response)

    @overload
    def chat_completion(  # type: ignore
        self,
        messages: List[Union[Dict, ChatCompletionInputMessage]],
        *,
        model: Optional[str] = None,
        stream: Literal[False] = False,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[List[float]] = None,
        logprobs: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[ChatCompletionInputGrammarType] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stream_options: Optional[ChatCompletionInputStreamOptions] = None,
        temperature: Optional[float] = None,
        tool_choice: Optional[Union[ChatCompletionInputToolChoiceClass, "ChatCompletionInputToolChoiceEnum"]] = None,
        tool_prompt: Optional[str] = None,
        tools: Optional[List[ChatCompletionInputTool]] = None,
        top_logprobs: Optional[int] = None,
        top_p: Optional[float] = None,
        extra_body: Optional[Dict] = None,
    ) -> ChatCompletionOutput: ...

    @overload
    def chat_completion(  # type: ignore
        self,
        messages: List[Union[Dict, ChatCompletionInputMessage]],
        *,
        model: Optional[str] = None,
        stream: Literal[True] = True,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[List[float]] = None,
        logprobs: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[ChatCompletionInputGrammarType] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stream_options: Optional[ChatCompletionInputStreamOptions] = None,
        temperature: Optional[float] = None,
        tool_choice: Optional[Union[ChatCompletionInputToolChoiceClass, "ChatCompletionInputToolChoiceEnum"]] = None,
        tool_prompt: Optional[str] = None,
        tools: Optional[List[ChatCompletionInputTool]] = None,
        top_logprobs: Optional[int] = None,
        top_p: Optional[float] = None,
        extra_body: Optional[Dict] = None,
    ) -> Iterable[ChatCompletionStreamOutput]: ...

    @overload
    def chat_completion(
        self,
        messages: List[Union[Dict, ChatCompletionInputMessage]],
        *,
        model: Optional[str] = None,
        stream: bool = False,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[List[float]] = None,
        logprobs: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[ChatCompletionInputGrammarType] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stream_options: Optional[ChatCompletionInputStreamOptions] = None,
        temperature: Optional[float] = None,
        tool_choice: Optional[Union[ChatCompletionInputToolChoiceClass, "ChatCompletionInputToolChoiceEnum"]] = None,
        tool_prompt: Optional[str] = None,
        tools: Optional[List[ChatCompletionInputTool]] = None,
        top_logprobs: Optional[int] = None,
        top_p: Optional[float] = None,
        extra_body: Optional[Dict] = None,
    ) -> Union[ChatCompletionOutput, Iterable[ChatCompletionStreamOutput]]: ...

    def chat_completion(
        self,
        messages: List[Union[Dict, ChatCompletionInputMessage]],
        *,
        model: Optional[str] = None,
        stream: bool = False,
        # Parameters from ChatCompletionInput (handled manually)
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[List[float]] = None,
        logprobs: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[ChatCompletionInputGrammarType] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stream_options: Optional[ChatCompletionInputStreamOptions] = None,
        temperature: Optional[float] = None,
        tool_choice: Optional[Union[ChatCompletionInputToolChoiceClass, "ChatCompletionInputToolChoiceEnum"]] = None,
        tool_prompt: Optional[str] = None,
        tools: Optional[List[ChatCompletionInputTool]] = None,
        top_logprobs: Optional[int] = None,
        top_p: Optional[float] = None,
        extra_body: Optional[Dict] = None,
    ) -> Union[ChatCompletionOutput, Iterable[ChatCompletionStreamOutput]]:
        """
        A method for completing conversations using a specified language model.

        <Tip>

        The `client.chat_completion` method is aliased as `client.chat.completions.create` for compatibility with OpenAI's client.
        Inputs and outputs are strictly the same and using either syntax will yield the same results.
        Check out the [Inference guide](https://huggingface.co/docs/huggingface_hub/guides/inference#openai-compatibility)
        for more details about OpenAI's compatibility.

        </Tip>

        <Tip>
        You can pass provider-specific parameters to the model by using the `extra_body` argument.
        </Tip>

        Args:
            messages (List of [`ChatCompletionInputMessage`]):
                Conversation history consisting of roles and content pairs.
            model (`str`, *optional*):
                The model to use for chat-completion. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. If not provided, the default recommended model for chat-based text-generation will be used.
                See https://huggingface.co/tasks/text-generation for more details.
                If `model` is a model ID, it is passed to the server as the `model` parameter. If you want to define a
                custom URL while setting `model` in the request payload, you must set `base_url` when initializing [`InferenceClient`].
            frequency_penalty (`float`, *optional*):
                Penalizes new tokens based on their existing frequency
                in the text so far. Range: [-2.0, 2.0]. Defaults to 0.0.
            logit_bias (`List[float]`, *optional*):
                Adjusts the likelihood of specific tokens appearing in the generated output.
            logprobs (`bool`, *optional*):
                Whether to return log probabilities of the output tokens or not. If true, returns the log
                probabilities of each output token returned in the content of message.
            max_tokens (`int`, *optional*):
                Maximum number of tokens allowed in the response. Defaults to 100.
            n (`int`, *optional*):
                The number of completions to generate for each prompt.
            presence_penalty (`float`, *optional*):
                Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the
                text so far, increasing the model's likelihood to talk about new topics.
            response_format ([`ChatCompletionInputGrammarType`], *optional*):
                Grammar constraints. Can be either a JSONSchema or a regex.
            seed (Optional[`int`], *optional*):
                Seed for reproducible control flow. Defaults to None.
            stop (`List[str]`, *optional*):
                Up to four strings which trigger the end of the response.
                Defaults to None.
            stream (`bool`, *optional*):
                Enable realtime streaming of responses. Defaults to False.
            stream_options ([`ChatCompletionInputStreamOptions`], *optional*):
                Options for streaming completions.
            temperature (`float`, *optional*):
                Controls randomness of the generations. Lower values ensure
                less random completions. Range: [0, 2]. Defaults to 1.0.
            top_logprobs (`int`, *optional*):
                An integer between 0 and 5 specifying the number of most likely tokens to return at each token
                position, each with an associated log probability. logprobs must be set to true if this parameter is
                used.
            top_p (`float`, *optional*):
                Fraction of the most likely next words to sample from.
                Must be between 0 and 1. Defaults to 1.0.
            tool_choice ([`ChatCompletionInputToolChoiceClass`] or [`ChatCompletionInputToolChoiceEnum`], *optional*):
                The tool to use for the completion. Defaults to "auto".
            tool_prompt (`str`, *optional*):
                A prompt to be appended before the tools.
            tools (List of [`ChatCompletionInputTool`], *optional*):
                A list of tools the model may call. Currently, only functions are supported as a tool. Use this to
                provide a list of functions the model may generate JSON inputs for.
            extra_body (`Dict`, *optional*):
                Additional provider-specific parameters to pass to the model. Refer to the provider's documentation
                for supported parameters.
        Returns:
            [`ChatCompletionOutput`] or Iterable of [`ChatCompletionStreamOutput`]:
            Generated text returned from the server:
            - if `stream=False`, the generated text is returned as a [`ChatCompletionOutput`] (default).
            - if `stream=True`, the generated text is returned token by token as a sequence of [`ChatCompletionStreamOutput`].

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:

        ```py
        >>> from huggingface_hub import InferenceClient
        >>> messages = [{"role": "user", "content": "What is the capital of France?"}]
        >>> client = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct")
        >>> client.chat_completion(messages, max_tokens=100)
        ChatCompletionOutput(
            choices=[
                ChatCompletionOutputComplete(
                    finish_reason='eos_token',
                    index=0,
                    message=ChatCompletionOutputMessage(
                        role='assistant',
                        content='The capital of France is Paris.',
                        name=None,
                        tool_calls=None
                    ),
                    logprobs=None
                )
            ],
            created=1719907176,
            id='',
            model='meta-llama/Meta-Llama-3-8B-Instruct',
            object='text_completion',
            system_fingerprint='2.0.4-sha-f426a33',
            usage=ChatCompletionOutputUsage(
                completion_tokens=8,
                prompt_tokens=17,
                total_tokens=25
            )
        )
        ```

        Example using streaming:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> messages = [{"role": "user", "content": "What is the capital of France?"}]
        >>> client = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct")
        >>> for token in client.chat_completion(messages, max_tokens=10, stream=True):
        ...     print(token)
        ChatCompletionStreamOutput(choices=[ChatCompletionStreamOutputChoice(delta=ChatCompletionStreamOutputDelta(content='The', role='assistant'), index=0, finish_reason=None)], created=1710498504)
        ChatCompletionStreamOutput(choices=[ChatCompletionStreamOutputChoice(delta=ChatCompletionStreamOutputDelta(content=' capital', role='assistant'), index=0, finish_reason=None)], created=1710498504)
        (...)
        ChatCompletionStreamOutput(choices=[ChatCompletionStreamOutputChoice(delta=ChatCompletionStreamOutputDelta(content=' may', role='assistant'), index=0, finish_reason=None)], created=1710498504)
        ```

        Example using OpenAI's syntax:
        ```py
        # instead of `from openai import OpenAI`
        from huggingface_hub import InferenceClient

        # instead of `client = OpenAI(...)`
        client = InferenceClient(
            base_url=...,
            api_key=...,
        )

        output = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Count to 10"},
            ],
            stream=True,
            max_tokens=1024,
        )

        for chunk in output:
            print(chunk.choices[0].delta.content)
        ```

        Example using a third-party provider directly with extra (provider-specific) parameters. Usage will be billed on your Together AI account.
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient(
        ...     provider="together",  # Use Together AI provider
        ...     api_key="<together_api_key>",  # Pass your Together API key directly
        ... )
        >>> client.chat_completion(
        ...     model="meta-llama/Meta-Llama-3-8B-Instruct",
        ...     messages=[{"role": "user", "content": "What is the capital of France?"}],
        ...     extra_body={"safety_model": "Meta-Llama/Llama-Guard-7b"},
        ... )
        ```

        Example using a third-party provider through Hugging Face Routing. Usage will be billed on your Hugging Face account.
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient(
        ...     provider="sambanova",  # Use Sambanova provider
        ...     api_key="hf_...",  # Pass your HF token
        ... )
        >>> client.chat_completion(
        ...     model="meta-llama/Meta-Llama-3-8B-Instruct",
        ...     messages=[{"role": "user", "content": "What is the capital of France?"}],
        ... )
        ```

        Example using Image + Text as input:
        ```py
        >>> from huggingface_hub import InferenceClient

        # provide a remote URL
        >>> image_url ="https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        # or a base64-encoded image
        >>> image_path = "/path/to/image.jpeg"
        >>> with open(image_path, "rb") as f:
        ...     base64_image = base64.b64encode(f.read()).decode("utf-8")
        >>> image_url = f"data:image/jpeg;base64,{base64_image}"

        >>> client = InferenceClient("meta-llama/Llama-3.2-11B-Vision-Instruct")
        >>> output = client.chat.completions.create(
        ...     messages=[
        ...         {
        ...             "role": "user",
        ...             "content": [
        ...                 {
        ...                     "type": "image_url",
        ...                     "image_url": {"url": image_url},
        ...                 },
        ...                 {
        ...                     "type": "text",
        ...                     "text": "Describe this image in one sentence.",
        ...                 },
        ...             ],
        ...         },
        ...     ],
        ... )
        >>> output
        The image depicts the iconic Statue of Liberty situated in New York Harbor, New York, on a clear day.
        ```

        Example using tools:
        ```py
        >>> client = InferenceClient("meta-llama/Meta-Llama-3-70B-Instruct")
        >>> messages = [
        ...     {
        ...         "role": "system",
        ...         "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.",
        ...     },
        ...     {
        ...         "role": "user",
        ...         "content": "What's the weather like the next 3 days in San Francisco, CA?",
        ...     },
        ... ]
        >>> tools = [
        ...     {
        ...         "type": "function",
        ...         "function": {
        ...             "name": "get_current_weather",
        ...             "description": "Get the current weather",
        ...             "parameters": {
        ...                 "type": "object",
        ...                 "properties": {
        ...                     "location": {
        ...                         "type": "string",
        ...                         "description": "The city and state, e.g. San Francisco, CA",
        ...                     },
        ...                     "format": {
        ...                         "type": "string",
        ...                         "enum": ["celsius", "fahrenheit"],
        ...                         "description": "The temperature unit to use. Infer this from the users location.",
        ...                     },
        ...                 },
        ...                 "required": ["location", "format"],
        ...             },
        ...         },
        ...     },
        ...     {
        ...         "type": "function",
        ...         "function": {
        ...             "name": "get_n_day_weather_forecast",
        ...             "description": "Get an N-day weather forecast",
        ...             "parameters": {
        ...                 "type": "object",
        ...                 "properties": {
        ...                     "location": {
        ...                         "type": "string",
        ...                         "description": "The city and state, e.g. San Francisco, CA",
        ...                     },
        ...                     "format": {
        ...                         "type": "string",
        ...                         "enum": ["celsius", "fahrenheit"],
        ...                         "description": "The temperature unit to use. Infer this from the users location.",
        ...                     },
        ...                     "num_days": {
        ...                         "type": "integer",
        ...                         "description": "The number of days to forecast",
        ...                     },
        ...                 },
        ...                 "required": ["location", "format", "num_days"],
        ...             },
        ...         },
        ...     },
        ... ]

        >>> response = client.chat_completion(
        ...     model="meta-llama/Meta-Llama-3-70B-Instruct",
        ...     messages=messages,
        ...     tools=tools,
        ...     tool_choice="auto",
        ...     max_tokens=500,
        ... )
        >>> response.choices[0].message.tool_calls[0].function
        ChatCompletionOutputFunctionDefinition(
            arguments={
                'location': 'San Francisco, CA',
                'format': 'fahrenheit',
                'num_days': 3
            },
            name='get_n_day_weather_forecast',
            description=None
        )
        ```

        Example using response_format:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient("meta-llama/Meta-Llama-3-70B-Instruct")
        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": "I saw a puppy a cat and a raccoon during my bike ride in the park. What did I saw and when?",
        ...     },
        ... ]
        >>> response_format = {
        ...     "type": "json",
        ...     "value": {
        ...         "properties": {
        ...             "location": {"type": "string"},
        ...             "activity": {"type": "string"},
        ...             "animals_seen": {"type": "integer", "minimum": 1, "maximum": 5},
        ...             "animals": {"type": "array", "items": {"type": "string"}},
        ...         },
        ...         "required": ["location", "activity", "animals_seen", "animals"],
        ...     },
        ... }
        >>> response = client.chat_completion(
        ...     messages=messages,
        ...     response_format=response_format,
        ...     max_tokens=500,
        ... )
        >>> response.choices[0].message.content
        '{\n\n"activity": "bike ride",\n"animals": ["puppy", "cat", "raccoon"],\n"animals_seen": 3,\n"location": "park"}'
        ```
        """
        # Since `chat_completion(..., model=xxx)` is also a payload parameter for the server, we need to handle 'model' differently.
        # `self.model` takes precedence over 'model' argument for building URL.
        # `model` takes precedence for payload value.
        model_id_or_url = self.model or model
        payload_model = model or self.model

        # Get the provider helper
        provider_helper = get_provider_helper(
            self.provider,
            task="conversational",
            model=model_id_or_url
            if model_id_or_url is not None and model_id_or_url.startswith(("http://", "https://"))
            else payload_model,
        )

        # Prepare the payload
        parameters = {
            "model": payload_model,
            "frequency_penalty": frequency_penalty,
            "logit_bias": logit_bias,
            "logprobs": logprobs,
            "max_tokens": max_tokens,
            "n": n,
            "presence_penalty": presence_penalty,
            "response_format": response_format,
            "seed": seed,
            "stop": stop,
            "temperature": temperature,
            "tool_choice": tool_choice,
            "tool_prompt": tool_prompt,
            "tools": tools,
            "top_logprobs": top_logprobs,
            "top_p": top_p,
            "stream": stream,
            "stream_options": stream_options,
            **(extra_body or {}),
        }
        request_parameters = provider_helper.prepare_request(
            inputs=messages,
            parameters=parameters,
            headers=self.headers,
            model=model_id_or_url,
            api_key=self.token,
        )
        data = self._inner_post(request_parameters, stream=stream)

        if stream:
            return _stream_chat_completion_response(data)  # type: ignore[arg-type]

        return ChatCompletionOutput.parse_obj_as_instance(data)  # type: ignore[arg-type]

    def document_question_answering(
        self,
        image: ContentT,
        question: str,
        *,
        model: Optional[str] = None,
        doc_stride: Optional[int] = None,
        handle_impossible_answer: Optional[bool] = None,
        lang: Optional[str] = None,
        max_answer_len: Optional[int] = None,
        max_question_len: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        top_k: Optional[int] = None,
        word_boxes: Optional[List[Union[List[float], str]]] = None,
    ) -> List[DocumentQuestionAnsweringOutputElement]:
        """
        Answer questions on document images.

        Args:
            image (`Union[str, Path, bytes, BinaryIO]`):
                The input image for the context. It can be raw bytes, an image file, or a URL to an online image.
            question (`str`):
                Question to be answered.
            model (`str`, *optional*):
                The model to use for the document question answering task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended document question answering model will be used.
                Defaults to None.
            doc_stride (`int`, *optional*):
                If the words in the document are too long to fit with the question for the model, it will be split in
                several chunks with some overlap. This argument controls the size of that overlap.
            handle_impossible_answer (`bool`, *optional*):
                Whether to accept impossible as an answer
            lang (`str`, *optional*):
                Language to use while running OCR. Defaults to english.
            max_answer_len (`int`, *optional*):
                The maximum length of predicted answers (e.g., only answers with a shorter length are considered).
            max_question_len (`int`, *optional*):
                The maximum length of the question after tokenization. It will be truncated if needed.
            max_seq_len (`int`, *optional*):
                The maximum length of the total sentence (context + question) in tokens of each chunk passed to the
                model. The context will be split in several chunks (using doc_stride as overlap) if needed.
            top_k (`int`, *optional*):
                The number of answers to return (will be chosen by order of likelihood). Can return less than top_k
                answers if there are not enough options available within the context.
            word_boxes (`List[Union[List[float], str`, *optional*):
                A list of words and bounding boxes (normalized 0->1000). If provided, the inference will skip the OCR
                step and use the provided bounding boxes instead.
        Returns:
            `List[DocumentQuestionAnsweringOutputElement]`: a list of [`DocumentQuestionAnsweringOutputElement`] items containing the predicted label, associated probability, word ids, and page number.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.


        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.document_question_answering(image="https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png", question="What is the invoice number?")
        [DocumentQuestionAnsweringOutputElement(answer='us-001', end=16, score=0.9999666213989258, start=16)]
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="document-question-answering", model=model_id)
        inputs: Dict[str, Any] = {"question": question, "image": _b64_encode(image)}
        request_parameters = provider_helper.prepare_request(
            inputs=inputs,
            parameters={
                "doc_stride": doc_stride,
                "handle_impossible_answer": handle_impossible_answer,
                "lang": lang,
                "max_answer_len": max_answer_len,
                "max_question_len": max_question_len,
                "max_seq_len": max_seq_len,
                "top_k": top_k,
                "word_boxes": word_boxes,
            },
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        return DocumentQuestionAnsweringOutputElement.parse_obj_as_list(response)

    def feature_extraction(
        self,
        text: str,
        *,
        normalize: Optional[bool] = None,
        prompt_name: Optional[str] = None,
        truncate: Optional[bool] = None,
        truncation_direction: Optional[Literal["Left", "Right"]] = None,
        model: Optional[str] = None,
    ) -> "np.ndarray":
        """
        Generate embeddings for a given text.

        Args:
            text (`str`):
                The text to embed.
            model (`str`, *optional*):
                The model to use for the feature extraction task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended feature extraction model will be used.
                Defaults to None.
            normalize (`bool`, *optional*):
                Whether to normalize the embeddings or not.
                Only available on server powered by Text-Embedding-Inference.
            prompt_name (`str`, *optional*):
                The name of the prompt that should be used by for encoding. If not set, no prompt will be applied.
                Must be a key in the `Sentence Transformers` configuration `prompts` dictionary.
                For example if ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ",...},
                then the sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?"
                because the prompt text will be prepended before any text to encode.
            truncate (`bool`, *optional*):
                Whether to truncate the embeddings or not.
                Only available on server powered by Text-Embedding-Inference.
            truncation_direction (`Literal["Left", "Right"]`, *optional*):
                Which side of the input should be truncated when `truncate=True` is passed.

        Returns:
            `np.ndarray`: The embedding representing the input text as a float32 numpy array.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.feature_extraction("Hi, who are you?")
        array([[ 2.424802  ,  2.93384   ,  1.1750331 , ...,  1.240499, -0.13776633, -0.7889173 ],
        [-0.42943227, -0.6364878 , -1.693462  , ...,  0.41978157, -2.4336355 ,  0.6162071 ],
        ...,
        [ 0.28552425, -0.928395  , -1.2077185 , ...,  0.76810825, -2.1069427 ,  0.6236161 ]], dtype=float32)
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="feature-extraction", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=text,
            parameters={
                "normalize": normalize,
                "prompt_name": prompt_name,
                "truncate": truncate,
                "truncation_direction": truncation_direction,
            },
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        np = _import_numpy()
        return np.array(provider_helper.get_response(response), dtype="float32")

    def fill_mask(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        targets: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[FillMaskOutputElement]:
        """
        Fill in a hole with a missing word (token to be precise).

        Args:
            text (`str`):
                a string to be filled from, must contain the [MASK] token (check model card for exact name of the mask).
            model (`str`, *optional*):
                The model to use for the fill mask task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended fill mask model will be used.
            targets (`List[str`, *optional*):
                When passed, the model will limit the scores to the passed targets instead of looking up in the whole
                vocabulary. If the provided targets are not in the model vocab, they will be tokenized and the first
                resulting token will be used (with a warning, and that might be slower).
            top_k (`int`, *optional*):
                When passed, overrides the number of predictions to return.
        Returns:
            `List[FillMaskOutputElement]`: a list of [`FillMaskOutputElement`] items containing the predicted label, associated
            probability, token reference, and completed text.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.fill_mask("The goal of life is <mask>.")
        [
            FillMaskOutputElement(score=0.06897063553333282, token=11098, token_str=' happiness', sequence='The goal of life is happiness.'),
            FillMaskOutputElement(score=0.06554922461509705, token=45075, token_str=' immortality', sequence='The goal of life is immortality.')
        ]
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="fill-mask", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=text,
            parameters={"targets": targets, "top_k": top_k},
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        return FillMaskOutputElement.parse_obj_as_list(response)

    def image_classification(
        self,
        image: ContentT,
        *,
        model: Optional[str] = None,
        function_to_apply: Optional["ImageClassificationOutputTransform"] = None,
        top_k: Optional[int] = None,
    ) -> List[ImageClassificationOutputElement]:
        """
        Perform image classification on the given image using the specified model.

        Args:
            image (`Union[str, Path, bytes, BinaryIO, PIL.Image.Image]`):
                The image to classify. It can be raw bytes, an image file, a URL to an online image, or a PIL Image.
            model (`str`, *optional*):
                The model to use for image classification. Can be a model ID hosted on the Hugging Face Hub or a URL to a
                deployed Inference Endpoint. If not provided, the default recommended model for image classification will be used.
            function_to_apply (`"ImageClassificationOutputTransform"`, *optional*):
                The function to apply to the model outputs in order to retrieve the scores.
            top_k (`int`, *optional*):
                When specified, limits the output to the top K most probable classes.
        Returns:
            `List[ImageClassificationOutputElement]`: a list of [`ImageClassificationOutputElement`] items containing the predicted label and associated probability.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.image_classification("https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg")
        [ImageClassificationOutputElement(label='Blenheim spaniel', score=0.9779096841812134), ...]
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="image-classification", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=image,
            parameters={"function_to_apply": function_to_apply, "top_k": top_k},
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        return ImageClassificationOutputElement.parse_obj_as_list(response)

    def image_segmentation(
        self,
        image: ContentT,
        *,
        model: Optional[str] = None,
        mask_threshold: Optional[float] = None,
        overlap_mask_area_threshold: Optional[float] = None,
        subtask: Optional["ImageSegmentationSubtask"] = None,
        threshold: Optional[float] = None,
    ) -> List[ImageSegmentationOutputElement]:
        """
        Perform image segmentation on the given image using the specified model.

        <Tip warning={true}>

        You must have `PIL` installed if you want to work with images (`pip install Pillow`).

        </Tip>

        Args:
            image (`Union[str, Path, bytes, BinaryIO, PIL.Image.Image]`):
                The image to segment. It can be raw bytes, an image file, a URL to an online image, or a PIL Image.
            model (`str`, *optional*):
                The model to use for image segmentation. Can be a model ID hosted on the Hugging Face Hub or a URL to a
                deployed Inference Endpoint. If not provided, the default recommended model for image segmentation will be used.
            mask_threshold (`float`, *optional*):
                Threshold to use when turning the predicted masks into binary values.
            overlap_mask_area_threshold (`float`, *optional*):
                Mask overlap threshold to eliminate small, disconnected segments.
            subtask (`"ImageSegmentationSubtask"`, *optional*):
                Segmentation task to be performed, depending on model capabilities.
            threshold (`float`, *optional*):
                Probability threshold to filter out predicted masks.
        Returns:
            `List[ImageSegmentationOutputElement]`: A list of [`ImageSegmentationOutputElement`] items containing the segmented masks and associated attributes.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.image_segmentation("cat.jpg")
        [ImageSegmentationOutputElement(score=0.989008, label='LABEL_184', mask=<PIL.PngImagePlugin.PngImageFile image mode=L size=400x300 at 0x7FDD2B129CC0>), ...]
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="image-segmentation", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=image,
            parameters={
                "mask_threshold": mask_threshold,
                "overlap_mask_area_threshold": overlap_mask_area_threshold,
                "subtask": subtask,
                "threshold": threshold,
            },
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        output = ImageSegmentationOutputElement.parse_obj_as_list(response)
        for item in output:
            item.mask = _b64_to_image(item.mask)  # type: ignore [assignment]
        return output

    def image_to_image(
        self,
        image: ContentT,
        prompt: Optional[str] = None,
        *,
        negative_prompt: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        model: Optional[str] = None,
        target_size: Optional[ImageToImageTargetSize] = None,
        **kwargs,
    ) -> "Image":
        """
        Perform image-to-image translation using a specified model.

        <Tip warning={true}>

        You must have `PIL` installed if you want to work with images (`pip install Pillow`).

        </Tip>

        Args:
            image (`Union[str, Path, bytes, BinaryIO, PIL.Image.Image]`):
                The input image for translation. It can be raw bytes, an image file, a URL to an online image, or a PIL Image.
            prompt (`str`, *optional*):
                The text prompt to guide the image generation.
            negative_prompt (`str`, *optional*):
                One prompt to guide what NOT to include in image generation.
            num_inference_steps (`int`, *optional*):
                For diffusion models. The number of denoising steps. More denoising steps usually lead to a higher
                quality image at the expense of slower inference.
            guidance_scale (`float`, *optional*):
                For diffusion models. A higher guidance scale value encourages the model to generate images closely
                linked to the text prompt at the expense of lower image quality.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. This parameter overrides the model defined at the instance level. Defaults to None.
            target_size (`ImageToImageTargetSize`, *optional*):
                The size in pixel of the output image.

        Returns:
            `Image`: The translated image.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> image = client.image_to_image("cat.jpg", prompt="turn the cat into a tiger")
        >>> image.save("tiger.jpg")
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="image-to-image", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=image,
            parameters={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "target_size": target_size,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                **kwargs,
            },
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        response = provider_helper.get_response(response, request_parameters)
        return _bytes_to_image(response)

    def image_to_text(self, image: ContentT, *, model: Optional[str] = None) -> ImageToTextOutput:
        """
        Takes an input image and return text.

        Models can have very different outputs depending on your use case (image captioning, optical character recognition
        (OCR), Pix2Struct, etc). Please have a look to the model card to learn more about a model's specificities.

        Args:
            image (`Union[str, Path, bytes, BinaryIO, PIL.Image.Image]`):
                The input image to caption. It can be raw bytes, an image file, a URL to an online image, or a PIL Image.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. This parameter overrides the model defined at the instance level. Defaults to None.

        Returns:
            [`ImageToTextOutput`]: The generated text.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.image_to_text("cat.jpg")
        'a cat standing in a grassy field '
        >>> client.image_to_text("https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg")
        'a dog laying on the grass next to a flower pot '
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="image-to-text", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=image,
            parameters={},
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        output = ImageToTextOutput.parse_obj(response)
        return output[0] if isinstance(output, list) else output

    def object_detection(
        self, image: ContentT, *, model: Optional[str] = None, threshold: Optional[float] = None
    ) -> List[ObjectDetectionOutputElement]:
        """
        Perform object detection on the given image using the specified model.

        <Tip warning={true}>

        You must have `PIL` installed if you want to work with images (`pip install Pillow`).

        </Tip>

        Args:
            image (`Union[str, Path, bytes, BinaryIO, PIL.Image.Image]`):
                The image to detect objects on. It can be raw bytes, an image file, a URL to an online image, or a PIL Image.
            model (`str`, *optional*):
                The model to use for object detection. Can be a model ID hosted on the Hugging Face Hub or a URL to a
                deployed Inference Endpoint. If not provided, the default recommended model for object detection (DETR) will be used.
            threshold (`float`, *optional*):
                The probability necessary to make a prediction.
        Returns:
            `List[ObjectDetectionOutputElement]`: A list of [`ObjectDetectionOutputElement`] items containing the bounding boxes and associated attributes.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.
            `ValueError`:
                If the request output is not a List.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.object_detection("people.jpg")
        [ObjectDetectionOutputElement(score=0.9486683011054993, label='person', box=ObjectDetectionBoundingBox(xmin=59, ymin=39, xmax=420, ymax=510)), ...]
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="object-detection", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=image,
            parameters={"threshold": threshold},
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        return ObjectDetectionOutputElement.parse_obj_as_list(response)

    def question_answering(
        self,
        question: str,
        context: str,
        *,
        model: Optional[str] = None,
        align_to_words: Optional[bool] = None,
        doc_stride: Optional[int] = None,
        handle_impossible_answer: Optional[bool] = None,
        max_answer_len: Optional[int] = None,
        max_question_len: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> Union[QuestionAnsweringOutputElement, List[QuestionAnsweringOutputElement]]:
        """
        Retrieve the answer to a question from a given text.

        Args:
            question (`str`):
                Question to be answered.
            context (`str`):
                The context of the question.
            model (`str`):
                The model to use for the question answering task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint.
            align_to_words (`bool`, *optional*):
                Attempts to align the answer to real words. Improves quality on space separated languages. Might hurt
                on non-space-separated languages (like Japanese or Chinese)
            doc_stride (`int`, *optional*):
                If the context is too long to fit with the question for the model, it will be split in several chunks
                with some overlap. This argument controls the size of that overlap.
            handle_impossible_answer (`bool`, *optional*):
                Whether to accept impossible as an answer.
            max_answer_len (`int`, *optional*):
                The maximum length of predicted answers (e.g., only answers with a shorter length are considered).
            max_question_len (`int`, *optional*):
                The maximum length of the question after tokenization. It will be truncated if needed.
            max_seq_len (`int`, *optional*):
                The maximum length of the total sentence (context + question) in tokens of each chunk passed to the
                model. The context will be split in several chunks (using docStride as overlap) if needed.
            top_k (`int`, *optional*):
                The number of answers to return (will be chosen by order of likelihood). Note that we return less than
                topk answers if there are not enough options available within the context.

        Returns:
            Union[`QuestionAnsweringOutputElement`, List[`QuestionAnsweringOutputElement`]]:
                When top_k is 1 or not provided, it returns a single `QuestionAnsweringOutputElement`.
                When top_k is greater than 1, it returns a list of `QuestionAnsweringOutputElement`.
        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.question_answering(question="What's my name?", context="My name is Clara and I live in Berkeley.")
        QuestionAnsweringOutputElement(answer='Clara', end=16, score=0.9326565265655518, start=11)
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="question-answering", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs={"question": question, "context": context},
            parameters={
                "align_to_words": align_to_words,
                "doc_stride": doc_stride,
                "handle_impossible_answer": handle_impossible_answer,
                "max_answer_len": max_answer_len,
                "max_question_len": max_question_len,
                "max_seq_len": max_seq_len,
                "top_k": top_k,
            },
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        # Parse the response as a single `QuestionAnsweringOutputElement` when top_k is 1 or not provided, or a list of `QuestionAnsweringOutputElement` to ensure backward compatibility.
        output = QuestionAnsweringOutputElement.parse_obj(response)
        return output

    def sentence_similarity(
        self, sentence: str, other_sentences: List[str], *, model: Optional[str] = None
    ) -> List[float]:
        """
        Compute the semantic similarity between a sentence and a list of other sentences by comparing their embeddings.

        Args:
            sentence (`str`):
                The main sentence to compare to others.
            other_sentences (`List[str]`):
                The list of sentences to compare to.
            model (`str`, *optional*):
                The model to use for the sentence similarity task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended sentence similarity model will be used.
                Defaults to None.

        Returns:
            `List[float]`: The embedding representing the input text.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.sentence_similarity(
        ...     "Machine learning is so easy.",
        ...     other_sentences=[
        ...         "Deep learning is so straightforward.",
        ...         "This is so difficult, like rocket science.",
        ...         "I can't believe how much I struggled with this.",
        ...     ],
        ... )
        [0.7785726189613342, 0.45876261591911316, 0.2906220555305481]
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="sentence-similarity", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs={"source_sentence": sentence, "sentences": other_sentences},
            parameters={},
            extra_payload={},
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        return _bytes_to_list(response)

    def summarization(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        clean_up_tokenization_spaces: Optional[bool] = None,
        generate_parameters: Optional[Dict[str, Any]] = None,
        truncation: Optional["SummarizationTruncationStrategy"] = None,
    ) -> SummarizationOutput:
        """
        Generate a summary of a given text using a specified model.

        Args:
            text (`str`):
                The input text to summarize.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. If not provided, the default recommended model for summarization will be used.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether to clean up the potential extra spaces in the text output.
            generate_parameters (`Dict[str, Any]`, *optional*):
                Additional parametrization of the text generation algorithm.
            truncation (`"SummarizationTruncationStrategy"`, *optional*):
                The truncation strategy to use.
        Returns:
            [`SummarizationOutput`]: The generated summary text.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.summarization("The Eiffel tower...")
        SummarizationOutput(generated_text="The Eiffel tower is one of the most famous landmarks in the world....")
        ```
        """
        parameters = {
            "clean_up_tokenization_spaces": clean_up_tokenization_spaces,
            "generate_parameters": generate_parameters,
            "truncation": truncation,
        }
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="summarization", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=text,
            parameters=parameters,
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        return SummarizationOutput.parse_obj_as_list(response)[0]

    def table_question_answering(
        self,
        table: Dict[str, Any],
        query: str,
        *,
        model: Optional[str] = None,
        padding: Optional["Padding"] = None,
        sequential: Optional[bool] = None,
        truncation: Optional[bool] = None,
    ) -> TableQuestionAnsweringOutputElement:
        """
        Retrieve the answer to a question from information given in a table.

        Args:
            table (`str`):
                A table of data represented as a dict of lists where entries are headers and the lists are all the
                values, all lists must have the same size.
            query (`str`):
                The query in plain text that you want to ask the table.
            model (`str`):
                The model to use for the table-question-answering task. Can be a model ID hosted on the Hugging Face
                Hub or a URL to a deployed Inference Endpoint.
            padding (`"Padding"`, *optional*):
                Activates and controls padding.
            sequential (`bool`, *optional*):
                Whether to do inference sequentially or as a batch. Batching is faster, but models like SQA require the
                inference to be done sequentially to extract relations within sequences, given their conversational
                nature.
            truncation (`bool`, *optional*):
                Activates and controls truncation.

        Returns:
            [`TableQuestionAnsweringOutputElement`]: a table question answering output containing the answer, coordinates, cells and the aggregator used.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> query = "How many stars does the transformers repository have?"
        >>> table = {"Repository": ["Transformers", "Datasets", "Tokenizers"], "Stars": ["36542", "4512", "3934"]}
        >>> client.table_question_answering(table, query, model="google/tapas-base-finetuned-wtq")
        TableQuestionAnsweringOutputElement(answer='36542', coordinates=[[0, 1]], cells=['36542'], aggregator='AVERAGE')
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="table-question-answering", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs={"query": query, "table": table},
            parameters={"model": model, "padding": padding, "sequential": sequential, "truncation": truncation},
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        return TableQuestionAnsweringOutputElement.parse_obj_as_instance(response)

    def tabular_classification(self, table: Dict[str, Any], *, model: Optional[str] = None) -> List[str]:
        """
        Classifying a target category (a group) based on a set of attributes.

        Args:
            table (`Dict[str, Any]`):
                Set of attributes to classify.
            model (`str`, *optional*):
                The model to use for the tabular classification task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended tabular classification model will be used.
                Defaults to None.

        Returns:
            `List`: a list of labels, one per row in the initial table.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> table = {
        ...     "fixed_acidity": ["7.4", "7.8", "10.3"],
        ...     "volatile_acidity": ["0.7", "0.88", "0.32"],
        ...     "citric_acid": ["0", "0", "0.45"],
        ...     "residual_sugar": ["1.9", "2.6", "6.4"],
        ...     "chlorides": ["0.076", "0.098", "0.073"],
        ...     "free_sulfur_dioxide": ["11", "25", "5"],
        ...     "total_sulfur_dioxide": ["34", "67", "13"],
        ...     "density": ["0.9978", "0.9968", "0.9976"],
        ...     "pH": ["3.51", "3.2", "3.23"],
        ...     "sulphates": ["0.56", "0.68", "0.82"],
        ...     "alcohol": ["9.4", "9.8", "12.6"],
        ... }
        >>> client.tabular_classification(table=table, model="julien-c/wine-quality")
        ["5", "5", "5"]
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="tabular-classification", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=None,
            extra_payload={"table": table},
            parameters={},
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        return _bytes_to_list(response)

    def tabular_regression(self, table: Dict[str, Any], *, model: Optional[str] = None) -> List[float]:
        """
        Predicting a numerical target value given a set of attributes/features in a table.

        Args:
            table (`Dict[str, Any]`):
                Set of attributes stored in a table. The attributes used to predict the target can be both numerical and categorical.
            model (`str`, *optional*):
                The model to use for the tabular regression task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended tabular regression model will be used.
                Defaults to None.

        Returns:
            `List`: a list of predicted numerical target values.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> table = {
        ...     "Height": ["11.52", "12.48", "12.3778"],
        ...     "Length1": ["23.2", "24", "23.9"],
        ...     "Length2": ["25.4", "26.3", "26.5"],
        ...     "Length3": ["30", "31.2", "31.1"],
        ...     "Species": ["Bream", "Bream", "Bream"],
        ...     "Width": ["4.02", "4.3056", "4.6961"],
        ... }
        >>> client.tabular_regression(table, model="scikit-learn/Fish-Weight")
        [110, 120, 130]
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="tabular-regression", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=None,
            parameters={},
            extra_payload={"table": table},
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        return _bytes_to_list(response)

    def text_classification(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        top_k: Optional[int] = None,
        function_to_apply: Optional["TextClassificationOutputTransform"] = None,
    ) -> List[TextClassificationOutputElement]:
        """
        Perform text classification (e.g. sentiment-analysis) on the given text.

        Args:
            text (`str`):
                A string to be classified.
            model (`str`, *optional*):
                The model to use for the text classification task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended text classification model will be used.
                Defaults to None.
            top_k (`int`, *optional*):
                When specified, limits the output to the top K most probable classes.
            function_to_apply (`"TextClassificationOutputTransform"`, *optional*):
                The function to apply to the model outputs in order to retrieve the scores.

        Returns:
            `List[TextClassificationOutputElement]`: a list of [`TextClassificationOutputElement`] items containing the predicted label and associated probability.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.text_classification("I like you")
        [
            TextClassificationOutputElement(label='POSITIVE', score=0.9998695850372314),
            TextClassificationOutputElement(label='NEGATIVE', score=0.0001304351753788069),
        ]
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="text-classification", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=text,
            parameters={
                "function_to_apply": function_to_apply,
                "top_k": top_k,
            },
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        return TextClassificationOutputElement.parse_obj_as_list(response)[0]  # type: ignore [return-value]

    @overload
    def text_generation(
        self,
        prompt: str,
        *,
        details: Literal[True],
        stream: Literal[True],
        model: Optional[str] = None,
        # Parameters from `TextGenerationInputGenerateParameters` (maintained manually)
        adapter_id: Optional[str] = None,
        best_of: Optional[int] = None,
        decoder_input_details: Optional[bool] = None,
        do_sample: Optional[bool] = None,
        frequency_penalty: Optional[float] = None,
        grammar: Optional[TextGenerationInputGrammarType] = None,
        max_new_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        return_full_text: Optional[bool] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stop_sequences: Optional[List[str]] = None,  # Deprecated, use `stop` instead
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_n_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        truncate: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: Optional[bool] = None,
    ) -> Iterable[TextGenerationStreamOutput]: ...

    @overload
    def text_generation(
        self,
        prompt: str,
        *,
        details: Literal[True],
        stream: Optional[Literal[False]] = None,
        model: Optional[str] = None,
        # Parameters from `TextGenerationInputGenerateParameters` (maintained manually)
        adapter_id: Optional[str] = None,
        best_of: Optional[int] = None,
        decoder_input_details: Optional[bool] = None,
        do_sample: Optional[bool] = None,
        frequency_penalty: Optional[float] = None,
        grammar: Optional[TextGenerationInputGrammarType] = None,
        max_new_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        return_full_text: Optional[bool] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stop_sequences: Optional[List[str]] = None,  # Deprecated, use `stop` instead
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_n_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        truncate: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: Optional[bool] = None,
    ) -> TextGenerationOutput: ...

    @overload
    def text_generation(
        self,
        prompt: str,
        *,
        details: Optional[Literal[False]] = None,
        stream: Literal[True],
        model: Optional[str] = None,
        # Parameters from `TextGenerationInputGenerateParameters` (maintained manually)
        adapter_id: Optional[str] = None,
        best_of: Optional[int] = None,
        decoder_input_details: Optional[bool] = None,
        do_sample: Optional[bool] = None,
        frequency_penalty: Optional[float] = None,
        grammar: Optional[TextGenerationInputGrammarType] = None,
        max_new_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        return_full_text: Optional[bool] = None,  # Manual default value
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stop_sequences: Optional[List[str]] = None,  # Deprecated, use `stop` instead
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_n_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        truncate: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: Optional[bool] = None,
    ) -> Iterable[str]: ...

    @overload
    def text_generation(
        self,
        prompt: str,
        *,
        details: Optional[Literal[False]] = None,
        stream: Optional[Literal[False]] = None,
        model: Optional[str] = None,
        # Parameters from `TextGenerationInputGenerateParameters` (maintained manually)
        adapter_id: Optional[str] = None,
        best_of: Optional[int] = None,
        decoder_input_details: Optional[bool] = None,
        do_sample: Optional[bool] = None,
        frequency_penalty: Optional[float] = None,
        grammar: Optional[TextGenerationInputGrammarType] = None,
        max_new_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        return_full_text: Optional[bool] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stop_sequences: Optional[List[str]] = None,  # Deprecated, use `stop` instead
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_n_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        truncate: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: Optional[bool] = None,
    ) -> str: ...

    @overload
    def text_generation(
        self,
        prompt: str,
        *,
        details: Optional[bool] = None,
        stream: Optional[bool] = None,
        model: Optional[str] = None,
        # Parameters from `TextGenerationInputGenerateParameters` (maintained manually)
        adapter_id: Optional[str] = None,
        best_of: Optional[int] = None,
        decoder_input_details: Optional[bool] = None,
        do_sample: Optional[bool] = None,
        frequency_penalty: Optional[float] = None,
        grammar: Optional[TextGenerationInputGrammarType] = None,
        max_new_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        return_full_text: Optional[bool] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stop_sequences: Optional[List[str]] = None,  # Deprecated, use `stop` instead
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_n_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        truncate: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: Optional[bool] = None,
    ) -> Union[str, TextGenerationOutput, Iterable[str], Iterable[TextGenerationStreamOutput]]: ...

    def text_generation(
        self,
        prompt: str,
        *,
        details: Optional[bool] = None,
        stream: Optional[bool] = None,
        model: Optional[str] = None,
        # Parameters from `TextGenerationInputGenerateParameters` (maintained manually)
        adapter_id: Optional[str] = None,
        best_of: Optional[int] = None,
        decoder_input_details: Optional[bool] = None,
        do_sample: Optional[bool] = None,
        frequency_penalty: Optional[float] = None,
        grammar: Optional[TextGenerationInputGrammarType] = None,
        max_new_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        return_full_text: Optional[bool] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stop_sequences: Optional[List[str]] = None,  # Deprecated, use `stop` instead
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_n_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        truncate: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: Optional[bool] = None,
    ) -> Union[str, TextGenerationOutput, Iterable[str], Iterable[TextGenerationStreamOutput]]:
        """
        Given a prompt, generate the following text.

        <Tip>

        If you want to generate a response from chat messages, you should use the [`InferenceClient.chat_completion`] method.
        It accepts a list of messages instead of a single text prompt and handles the chat templating for you.

        </Tip>

        Args:
            prompt (`str`):
                Input text.
            details (`bool`, *optional*):
                By default, text_generation returns a string. Pass `details=True` if you want a detailed output (tokens,
                probabilities, seed, finish reason, etc.). Only available for models running on with the
                `text-generation-inference` backend.
            stream (`bool`, *optional*):
                By default, text_generation returns the full generated text. Pass `stream=True` if you want a stream of
                tokens to be returned. Only available for models running on with the `text-generation-inference`
                backend.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. This parameter overrides the model defined at the instance level. Defaults to None.
            adapter_id (`str`, *optional*):
                Lora adapter id.
            best_of (`int`, *optional*):
                Generate best_of sequences and return the one if the highest token logprobs.
            decoder_input_details (`bool`, *optional*):
                Return the decoder input token logprobs and ids. You must set `details=True` as well for it to be taken
                into account. Defaults to `False`.
            do_sample (`bool`, *optional*):
                Activate logits sampling
            frequency_penalty (`float`, *optional*):
                Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in
                the text so far, decreasing the model's likelihood to repeat the same line verbatim.
            grammar ([`TextGenerationInputGrammarType`], *optional*):
                Grammar constraints. Can be either a JSONSchema or a regex.
            max_new_tokens (`int`, *optional*):
                Maximum number of generated tokens. Defaults to 100.
            repetition_penalty (`float`, *optional*):
                The parameter for repetition penalty. 1.0 means no penalty. See [this
                paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
            return_full_text (`bool`, *optional*):
                Whether to prepend the prompt to the generated text
            seed (`int`, *optional*):
                Random sampling seed
            stop (`List[str]`, *optional*):
                Stop generating tokens if a member of `stop` is generated.
            stop_sequences (`List[str]`, *optional*):
                Deprecated argument. Use `stop` instead.
            temperature (`float`, *optional*):
                The value used to module the logits distribution.
            top_n_tokens (`int`, *optional*):
                Return information about the `top_n_tokens` most likely tokens at each generation step, instead of
                just the sampled token.
            top_k (`int`, *optional`):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`, *optional`):
                If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
                higher are kept for generation.
            truncate (`int`, *optional`):
                Truncate inputs tokens to the given size.
            typical_p (`float`, *optional`):
                Typical Decoding mass
                See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information
            watermark (`bool`, *optional*):
                Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)

        Returns:
            `Union[str, TextGenerationOutput, Iterable[str], Iterable[TextGenerationStreamOutput]]`:
            Generated text returned from the server:
            - if `stream=False` and `details=False`, the generated text is returned as a `str` (default)
            - if `stream=True` and `details=False`, the generated text is returned token by token as a `Iterable[str]`
            - if `stream=False` and `details=True`, the generated text is returned with more details as a [`~huggingface_hub.TextGenerationOutput`]
            - if `details=True` and `stream=True`, the generated text is returned token by token as a iterable of [`~huggingface_hub.TextGenerationStreamOutput`]

        Raises:
            `ValidationError`:
                If input values are not valid. No HTTP call is made to the server.
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()

        # Case 1: generate text
        >>> client.text_generation("The huggingface_hub library is ", max_new_tokens=12)
        '100% open source and built to be easy to use.'

        # Case 2: iterate over the generated tokens. Useful for large generation.
        >>> for token in client.text_generation("The huggingface_hub library is ", max_new_tokens=12, stream=True):
        ...     print(token)
        100
        %
        open
        source
        and
        built
        to
        be
        easy
        to
        use
        .

        # Case 3: get more details about the generation process.
        >>> client.text_generation("The huggingface_hub library is ", max_new_tokens=12, details=True)
        TextGenerationOutput(
            generated_text='100% open source and built to be easy to use.',
            details=TextGenerationDetails(
                finish_reason='length',
                generated_tokens=12,
                seed=None,
                prefill=[
                    TextGenerationPrefillOutputToken(id=487, text='The', logprob=None),
                    TextGenerationPrefillOutputToken(id=53789, text=' hugging', logprob=-13.171875),
                    (...)
                    TextGenerationPrefillOutputToken(id=204, text=' ', logprob=-7.0390625)
                ],
                tokens=[
                    TokenElement(id=1425, text='100', logprob=-1.0175781, special=False),
                    TokenElement(id=16, text='%', logprob=-0.0463562, special=False),
                    (...)
                    TokenElement(id=25, text='.', logprob=-0.5703125, special=False)
                ],
                best_of_sequences=None
            )
        )

        # Case 4: iterate over the generated tokens with more details.
        # Last object is more complete, containing the full generated text and the finish reason.
        >>> for details in client.text_generation("The huggingface_hub library is ", max_new_tokens=12, details=True, stream=True):
        ...     print(details)
        ...
        TextGenerationStreamOutput(token=TokenElement(id=1425, text='100', logprob=-1.0175781, special=False), generated_text=None, details=None)
        TextGenerationStreamOutput(token=TokenElement(id=16, text='%', logprob=-0.0463562, special=False), generated_text=None, details=None)
        TextGenerationStreamOutput(token=TokenElement(id=1314, text=' open', logprob=-1.3359375, special=False), generated_text=None, details=None)
        TextGenerationStreamOutput(token=TokenElement(id=3178, text=' source', logprob=-0.28100586, special=False), generated_text=None, details=None)
        TextGenerationStreamOutput(token=TokenElement(id=273, text=' and', logprob=-0.5961914, special=False), generated_text=None, details=None)
        TextGenerationStreamOutput(token=TokenElement(id=3426, text=' built', logprob=-1.9423828, special=False), generated_text=None, details=None)
        TextGenerationStreamOutput(token=TokenElement(id=271, text=' to', logprob=-1.4121094, special=False), generated_text=None, details=None)
        TextGenerationStreamOutput(token=TokenElement(id=314, text=' be', logprob=-1.5224609, special=False), generated_text=None, details=None)
        TextGenerationStreamOutput(token=TokenElement(id=1833, text=' easy', logprob=-2.1132812, special=False), generated_text=None, details=None)
        TextGenerationStreamOutput(token=TokenElement(id=271, text=' to', logprob=-0.08520508, special=False), generated_text=None, details=None)
        TextGenerationStreamOutput(token=TokenElement(id=745, text=' use', logprob=-0.39453125, special=False), generated_text=None, details=None)
        TextGenerationStreamOutput(token=TokenElement(
            id=25,
            text='.',
            logprob=-0.5703125,
            special=False),
            generated_text='100% open source and built to be easy to use.',
            details=TextGenerationStreamOutputStreamDetails(finish_reason='length', generated_tokens=12, seed=None)
        )

        # Case 5: generate constrained output using grammar
        >>> response = client.text_generation(
        ...     prompt="I saw a puppy a cat and a raccoon during my bike ride in the park",
        ...     model="HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1",
        ...     max_new_tokens=100,
        ...     repetition_penalty=1.3,
        ...     grammar={
        ...         "type": "json",
        ...         "value": {
        ...             "properties": {
        ...                 "location": {"type": "string"},
        ...                 "activity": {"type": "string"},
        ...                 "animals_seen": {"type": "integer", "minimum": 1, "maximum": 5},
        ...                 "animals": {"type": "array", "items": {"type": "string"}},
        ...             },
        ...             "required": ["location", "activity", "animals_seen", "animals"],
        ...         },
        ...     },
        ... )
        >>> json.loads(response)
        {
            "activity": "bike riding",
            "animals": ["puppy", "cat", "raccoon"],
            "animals_seen": 3,
            "location": "park"
        }
        ```
        """
        if decoder_input_details and not details:
            warnings.warn(
                "`decoder_input_details=True` has been passed to the server but `details=False` is set meaning that"
                " the output from the server will be truncated."
            )
            decoder_input_details = False

        if stop_sequences is not None:
            warnings.warn(
                "`stop_sequences` is a deprecated argument for `text_generation` task"
                " and will be removed in version '0.28.0'. Use `stop` instead.",
                FutureWarning,
            )
        if stop is None:
            stop = stop_sequences  # use deprecated arg if provided

        # Build payload
        parameters = {
            "adapter_id": adapter_id,
            "best_of": best_of,
            "decoder_input_details": decoder_input_details,
            "details": details,
            "do_sample": do_sample,
            "frequency_penalty": frequency_penalty,
            "grammar": grammar,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "return_full_text": return_full_text,
            "seed": seed,
            "stop": stop,
            "temperature": temperature,
            "top_k": top_k,
            "top_n_tokens": top_n_tokens,
            "top_p": top_p,
            "truncate": truncate,
            "typical_p": typical_p,
            "watermark": watermark,
        }

        # Remove some parameters if not a TGI server
        unsupported_kwargs = _get_unsupported_text_generation_kwargs(model)
        if len(unsupported_kwargs) > 0:
            # The server does not support some parameters
            # => means it is not a TGI server
            # => remove unsupported parameters and warn the user

            ignored_parameters = []
            for key in unsupported_kwargs:
                if parameters.get(key):
                    ignored_parameters.append(key)
                parameters.pop(key, None)
            if len(ignored_parameters) > 0:
                warnings.warn(
                    "API endpoint/model for text-generation is not served via TGI. Ignoring following parameters:"
                    f" {', '.join(ignored_parameters)}.",
                    UserWarning,
                )
            if details:
                warnings.warn(
                    "API endpoint/model for text-generation is not served via TGI. Parameter `details=True` will"
                    " be ignored meaning only the generated text will be returned.",
                    UserWarning,
                )
                details = False
            if stream:
                raise ValueError(
                    "API endpoint/model for text-generation is not served via TGI. Cannot return output as a stream."
                    " Please pass `stream=False` as input."
                )

        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="text-generation", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=prompt,
            parameters=parameters,
            extra_payload={"stream": stream},
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )

        # Handle errors separately for more precise error messages
        try:
            bytes_output = self._inner_post(request_parameters, stream=stream or False)
        except HTTPError as e:
            match = MODEL_KWARGS_NOT_USED_REGEX.search(str(e))
            if isinstance(e, BadRequestError) and match:
                unused_params = [kwarg.strip("' ") for kwarg in match.group(1).split(",")]
                _set_unsupported_text_generation_kwargs(model, unused_params)
                return self.text_generation(  # type: ignore
                    prompt=prompt,
                    details=details,
                    stream=stream,
                    model=model_id,
                    adapter_id=adapter_id,
                    best_of=best_of,
                    decoder_input_details=decoder_input_details,
                    do_sample=do_sample,
                    frequency_penalty=frequency_penalty,
                    grammar=grammar,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=repetition_penalty,
                    return_full_text=return_full_text,
                    seed=seed,
                    stop=stop,
                    temperature=temperature,
                    top_k=top_k,
                    top_n_tokens=top_n_tokens,
                    top_p=top_p,
                    truncate=truncate,
                    typical_p=typical_p,
                    watermark=watermark,
                )
            raise_text_generation_error(e)

        # Parse output
        if stream:
            return _stream_text_generation_response(bytes_output, details)  # type: ignore

        data = _bytes_to_dict(bytes_output)  # type: ignore[arg-type]

        # Data can be a single element (dict) or an iterable of dicts where we select the first element of.
        if isinstance(data, list):
            data = data[0]
        response = provider_helper.get_response(data, request_parameters)
        return TextGenerationOutput.parse_obj_as_instance(response) if details else response["generated_text"]

    def text_to_image(
        self,
        prompt: str,
        *,
        negative_prompt: Optional[str] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        model: Optional[str] = None,
        scheduler: Optional[str] = None,
        seed: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> "Image":
        """
        Generate an image based on a given text using a specified model.

        <Tip warning={true}>

        You must have `PIL` installed if you want to work with images (`pip install Pillow`).

        </Tip>

        <Tip>
        You can pass provider-specific parameters to the model by using the `extra_body` argument.
        </Tip>

        Args:
            prompt (`str`):
                The prompt to generate an image from.
            negative_prompt (`str`, *optional*):
                One prompt to guide what NOT to include in image generation.
            height (`int`, *optional*):
                The height in pixels of the output image
            width (`int`, *optional*):
                The width in pixels of the output image
            num_inference_steps (`int`, *optional*):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                prompt, but values too high may cause saturation and other artifacts.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. If not provided, the default recommended text-to-image model will be used.
                Defaults to None.
            scheduler (`str`, *optional*):
                Override the scheduler with a compatible one.
            seed (`int`, *optional*):
                Seed for the random number generator.
            extra_body (`Dict[str, Any]`, *optional*):
                Additional provider-specific parameters to pass to the model. Refer to the provider's documentation
                for supported parameters.

        Returns:
            `Image`: The generated image.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()

        >>> image = client.text_to_image("An astronaut riding a horse on the moon.")
        >>> image.save("astronaut.png")

        >>> image = client.text_to_image(
        ...     "An astronaut riding a horse on the moon.",
        ...     negative_prompt="low resolution, blurry",
        ...     model="stabilityai/stable-diffusion-2-1",
        ... )
        >>> image.save("better_astronaut.png")
        ```
        Example using a third-party provider directly. Usage will be billed on your fal.ai account.
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient(
        ...     provider="fal-ai",  # Use fal.ai provider
        ...     api_key="fal-ai-api-key",  # Pass your fal.ai API key
        ... )
        >>> image = client.text_to_image(
        ...     "A majestic lion in a fantasy forest",
        ...     model="black-forest-labs/FLUX.1-schnell",
        ... )
        >>> image.save("lion.png")
        ```

        Example using a third-party provider through Hugging Face Routing. Usage will be billed on your Hugging Face account.
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient(
        ...     provider="replicate",  # Use replicate provider
        ...     api_key="hf_...",  # Pass your HF token
        ... )
        >>> image = client.text_to_image(
        ...     "An astronaut riding a horse on the moon.",
        ...     model="black-forest-labs/FLUX.1-dev",
        ... )
        >>> image.save("astronaut.png")
        ```

        Example using Replicate provider with extra parameters
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient(
        ...     provider="replicate",  # Use replicate provider
        ...     api_key="hf_...",  # Pass your HF token
        ... )
        >>> image = client.text_to_image(
        ...     "An astronaut riding a horse on the moon.",
        ...     model="black-forest-labs/FLUX.1-schnell",
        ...     extra_body={"output_quality": 100},
        ... )
        >>> image.save("astronaut.png")
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="text-to-image", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=prompt,
            parameters={
                "negative_prompt": negative_prompt,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "scheduler": scheduler,
                "seed": seed,
                **(extra_body or {}),
            },
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        response = provider_helper.get_response(response)
        return _bytes_to_image(response)

    def text_to_video(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        guidance_scale: Optional[float] = None,
        negative_prompt: Optional[List[str]] = None,
        num_frames: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        """
        Generate a video based on a given text.

        <Tip>
        You can pass provider-specific parameters to the model by using the `extra_body` argument.
        </Tip>

        Args:
            prompt (`str`):
                The prompt to generate a video from.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. If not provided, the default recommended text-to-video model will be used.
                Defaults to None.
            guidance_scale (`float`, *optional*):
                A higher guidance scale value encourages the model to generate videos closely linked to the text
                prompt, but values too high may cause saturation and other artifacts.
            negative_prompt (`List[str]`, *optional*):
                One or several prompt to guide what NOT to include in video generation.
            num_frames (`float`, *optional*):
                The num_frames parameter determines how many video frames are generated.
            num_inference_steps (`int`, *optional*):
                The number of denoising steps. More denoising steps usually lead to a higher quality video at the
                expense of slower inference.
            seed (`int`, *optional*):
                Seed for the random number generator.
            extra_body (`Dict[str, Any]`, *optional*):
                Additional provider-specific parameters to pass to the model. Refer to the provider's documentation
                for supported parameters.

        Returns:
            `bytes`: The generated video.

        Example:

        Example using a third-party provider directly. Usage will be billed on your fal.ai account.
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient(
        ...     provider="fal-ai",  # Using fal.ai provider
        ...     api_key="fal-ai-api-key",  # Pass your fal.ai API key
        ... )
        >>> video = client.text_to_video(
        ...     "A majestic lion running in a fantasy forest",
        ...     model="tencent/HunyuanVideo",
        ... )
        >>> with open("lion.mp4", "wb") as file:
        ...     file.write(video)
        ```

        Example using a third-party provider through Hugging Face Routing. Usage will be billed on your Hugging Face account.
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient(
        ...     provider="replicate",  # Using replicate provider
        ...     api_key="hf_...",  # Pass your HF token
        ... )
        >>> video = client.text_to_video(
        ...     "A cat running in a park",
        ...     model="genmo/mochi-1-preview",
        ... )
        >>> with open("cat.mp4", "wb") as file:
        ...     file.write(video)
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="text-to-video", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=prompt,
            parameters={
                "guidance_scale": guidance_scale,
                "negative_prompt": negative_prompt,
                "num_frames": num_frames,
                "num_inference_steps": num_inference_steps,
                "seed": seed,
                **(extra_body or {}),
            },
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        response = provider_helper.get_response(response, request_parameters)
        return response

    def text_to_speech(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[Union[bool, "TextToSpeechEarlyStoppingEnum"]] = None,
        epsilon_cutoff: Optional[float] = None,
        eta_cutoff: Optional[float] = None,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        min_length: Optional[int] = None,
        min_new_tokens: Optional[int] = None,
        num_beam_groups: Optional[int] = None,
        num_beams: Optional[int] = None,
        penalty_alpha: Optional[float] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        use_cache: Optional[bool] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        """
        Synthesize an audio of a voice pronouncing a given text.

        <Tip>
        You can pass provider-specific parameters to the model by using the `extra_body` argument.
        </Tip>

        Args:
            text (`str`):
                The text to synthesize.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. If not provided, the default recommended text-to-speech model will be used.
                Defaults to None.
            do_sample (`bool`, *optional*):
                Whether to use sampling instead of greedy decoding when generating new tokens.
            early_stopping (`Union[bool, "TextToSpeechEarlyStoppingEnum"]`, *optional*):
                Controls the stopping condition for beam-based methods.
            epsilon_cutoff (`float`, *optional*):
                If set to float strictly between 0 and 1, only tokens with a conditional probability greater than
                epsilon_cutoff will be sampled. In the paper, suggested values range from 3e-4 to 9e-4, depending on
                the size of the model. See [Truncation Sampling as Language Model
                Desmoothing](https://hf.co/papers/2210.15191) for more details.
            eta_cutoff (`float`, *optional*):
                Eta sampling is a hybrid of locally typical sampling and epsilon sampling. If set to float strictly
                between 0 and 1, a token is only considered if it is greater than either eta_cutoff or sqrt(eta_cutoff)
                * exp(-entropy(softmax(next_token_logits))). The latter term is intuitively the expected next token
                probability, scaled by sqrt(eta_cutoff). In the paper, suggested values range from 3e-4 to 2e-3,
                depending on the size of the model. See [Truncation Sampling as Language Model
                Desmoothing](https://hf.co/papers/2210.15191) for more details.
            max_length (`int`, *optional*):
                The maximum length (in tokens) of the generated text, including the input.
            max_new_tokens (`int`, *optional*):
                The maximum number of tokens to generate. Takes precedence over max_length.
            min_length (`int`, *optional*):
                The minimum length (in tokens) of the generated text, including the input.
            min_new_tokens (`int`, *optional*):
                The minimum number of tokens to generate. Takes precedence over min_length.
            num_beam_groups (`int`, *optional*):
                Number of groups to divide num_beams into in order to ensure diversity among different groups of beams.
                See [this paper](https://hf.co/papers/1610.02424) for more details.
            num_beams (`int`, *optional*):
                Number of beams to use for beam search.
            penalty_alpha (`float`, *optional*):
                The value balances the model confidence and the degeneration penalty in contrastive search decoding.
            temperature (`float`, *optional*):
                The value used to modulate the next token probabilities.
            top_k (`int`, *optional*):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`, *optional*):
                If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to
                top_p or higher are kept for generation.
            typical_p (`float`, *optional*):
                Local typicality measures how similar the conditional probability of predicting a target token next is
                to the expected conditional probability of predicting a random token next, given the partial text
                already generated. If set to float < 1, the smallest set of the most locally typical tokens with
                probabilities that add up to typical_p or higher are kept for generation. See [this
                paper](https://hf.co/papers/2202.00666) for more details.
            use_cache (`bool`, *optional*):
                Whether the model should use the past last key/values attentions to speed up decoding
            extra_body (`Dict[str, Any]`, *optional*):
                Additional provider-specific parameters to pass to the model. Refer to the provider's documentation
                for supported parameters.
        Returns:
            `bytes`: The generated audio.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from pathlib import Path
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()

        >>> audio = client.text_to_speech("Hello world")
        >>> Path("hello_world.flac").write_bytes(audio)
        ```

        Example using a third-party provider directly. Usage will be billed on your Replicate account.
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient(
        ...     provider="replicate",
        ...     api_key="your-replicate-api-key",  # Pass your Replicate API key directly
        ... )
        >>> audio = client.text_to_speech(
        ...     text="Hello world",
        ...     model="OuteAI/OuteTTS-0.3-500M",
        ... )
        >>> Path("hello_world.flac").write_bytes(audio)
        ```

        Example using a third-party provider through Hugging Face Routing. Usage will be billed on your Hugging Face account.
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient(
        ...     provider="replicate",
        ...     api_key="hf_...",  # Pass your HF token
        ... )
        >>> audio =client.text_to_speech(
        ...     text="Hello world",
        ...     model="OuteAI/OuteTTS-0.3-500M",
        ... )
        >>> Path("hello_world.flac").write_bytes(audio)
        ```
        Example using Replicate provider with extra parameters
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient(
        ...     provider="replicate",  # Use replicate provider
        ...     api_key="hf_...",  # Pass your HF token
        ... )
        >>> audio = client.text_to_speech(
        ...     "Hello, my name is Kororo, an awesome text-to-speech model.",
        ...     model="hexgrad/Kokoro-82M",
        ...     extra_body={"voice": "af_nicole"},
        ... )
        >>> Path("hello.flac").write_bytes(audio)
        ```

        Example music-gen using "YuE-s1-7B-anneal-en-cot" on fal.ai
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> lyrics = '''
        ... [verse]
        ... In the town where I was born
        ... Lived a man who sailed to sea
        ... And he told us of his life
        ... In the land of submarines
        ... So we sailed on to the sun
        ... 'Til we found a sea of green
        ... And we lived beneath the waves
        ... In our yellow submarine

        ... [chorus]
        ... We all live in a yellow submarine
        ... Yellow submarine, yellow submarine
        ... We all live in a yellow submarine
        ... Yellow submarine, yellow submarine
        ... '''
        >>> genres = "pavarotti-style tenor voice"
        >>> client = InferenceClient(
        ...     provider="fal-ai",
        ...     model="m-a-p/YuE-s1-7B-anneal-en-cot",
        ...     api_key=...,
        ... )
        >>> audio = client.text_to_speech(lyrics, extra_body={"genres": genres})
        >>> with open("output.mp3", "wb") as f:
        ...     f.write(audio)
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="text-to-speech", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=text,
            parameters={
                "do_sample": do_sample,
                "early_stopping": early_stopping,
                "epsilon_cutoff": epsilon_cutoff,
                "eta_cutoff": eta_cutoff,
                "max_length": max_length,
                "max_new_tokens": max_new_tokens,
                "min_length": min_length,
                "min_new_tokens": min_new_tokens,
                "num_beam_groups": num_beam_groups,
                "num_beams": num_beams,
                "penalty_alpha": penalty_alpha,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "typical_p": typical_p,
                "use_cache": use_cache,
                **(extra_body or {}),
            },
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        response = provider_helper.get_response(response)
        return response

    def token_classification(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        aggregation_strategy: Optional["TokenClassificationAggregationStrategy"] = None,
        ignore_labels: Optional[List[str]] = None,
        stride: Optional[int] = None,
    ) -> List[TokenClassificationOutputElement]:
        """
        Perform token classification on the given text.
        Usually used for sentence parsing, either grammatical, or Named Entity Recognition (NER) to understand keywords contained within text.

        Args:
            text (`str`):
                A string to be classified.
            model (`str`, *optional*):
                The model to use for the token classification task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended token classification model will be used.
                Defaults to None.
            aggregation_strategy (`"TokenClassificationAggregationStrategy"`, *optional*):
                The strategy used to fuse tokens based on model predictions
            ignore_labels (`List[str`, *optional*):
                A list of labels to ignore
            stride (`int`, *optional*):
                The number of overlapping tokens between chunks when splitting the input text.

        Returns:
            `List[TokenClassificationOutputElement]`: List of [`TokenClassificationOutputElement`] items containing the entity group, confidence score, word, start and end index.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.token_classification("My name is Sarah Jessica Parker but you can call me Jessica")
        [
            TokenClassificationOutputElement(
                entity_group='PER',
                score=0.9971321225166321,
                word='Sarah Jessica Parker',
                start=11,
                end=31,
            ),
            TokenClassificationOutputElement(
                entity_group='PER',
                score=0.9773476123809814,
                word='Jessica',
                start=52,
                end=59,
            )
        ]
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="token-classification", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=text,
            parameters={
                "aggregation_strategy": aggregation_strategy,
                "ignore_labels": ignore_labels,
                "stride": stride,
            },
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        return TokenClassificationOutputElement.parse_obj_as_list(response)

    def translation(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
        clean_up_tokenization_spaces: Optional[bool] = None,
        truncation: Optional["TranslationTruncationStrategy"] = None,
        generate_parameters: Optional[Dict[str, Any]] = None,
    ) -> TranslationOutput:
        """
        Convert text from one language to another.

        Check out https://huggingface.co/tasks/translation for more information on how to choose the best model for
        your specific use case. Source and target languages usually depend on the model.
        However, it is possible to specify source and target languages for certain models. If you are working with one of these models,
        you can use `src_lang` and `tgt_lang` arguments to pass the relevant information.

        Args:
            text (`str`):
                A string to be translated.
            model (`str`, *optional*):
                The model to use for the translation task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended translation model will be used.
                Defaults to None.
            src_lang (`str`, *optional*):
                The source language of the text. Required for models that can translate from multiple languages.
            tgt_lang (`str`, *optional*):
                Target language to translate to. Required for models that can translate to multiple languages.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether to clean up the potential extra spaces in the text output.
            truncation (`"TranslationTruncationStrategy"`, *optional*):
                The truncation strategy to use.
            generate_parameters (`Dict[str, Any]`, *optional*):
                Additional parametrization of the text generation algorithm.

        Returns:
            [`TranslationOutput`]: The generated translated text.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.
            `ValueError`:
                If only one of the `src_lang` and `tgt_lang` arguments are provided.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.translation("My name is Wolfgang and I live in Berlin")
        'Mein Name ist Wolfgang und ich lebe in Berlin.'
        >>> client.translation("My name is Wolfgang and I live in Berlin", model="Helsinki-NLP/opus-mt-en-fr")
        TranslationOutput(translation_text='Je m'appelle Wolfgang et je vis à Berlin.')
        ```

        Specifying languages:
        ```py
        >>> client.translation("My name is Sarah Jessica Parker but you can call me Jessica", model="facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="fr_XX")
        "Mon nom est Sarah Jessica Parker mais vous pouvez m'appeler Jessica"
        ```
        """
        # Throw error if only one of `src_lang` and `tgt_lang` was given
        if src_lang is not None and tgt_lang is None:
            raise ValueError("You cannot specify `src_lang` without specifying `tgt_lang`.")

        if src_lang is None and tgt_lang is not None:
            raise ValueError("You cannot specify `tgt_lang` without specifying `src_lang`.")

        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="translation", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=text,
            parameters={
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "clean_up_tokenization_spaces": clean_up_tokenization_spaces,
                "truncation": truncation,
                "generate_parameters": generate_parameters,
            },
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        return TranslationOutput.parse_obj_as_list(response)[0]

    def visual_question_answering(
        self,
        image: ContentT,
        question: str,
        *,
        model: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[VisualQuestionAnsweringOutputElement]:
        """
        Answering open-ended questions based on an image.

        Args:
            image (`Union[str, Path, bytes, BinaryIO, PIL.Image.Image]`):
                The input image for the context. It can be raw bytes, an image file, a URL to an online image, or a PIL Image.
            question (`str`):
                Question to be answered.
            model (`str`, *optional*):
                The model to use for the visual question answering task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended visual question answering model will be used.
                Defaults to None.
            top_k (`int`, *optional*):
                The number of answers to return (will be chosen by order of likelihood). Note that we return less than
                topk answers if there are not enough options available within the context.
        Returns:
            `List[VisualQuestionAnsweringOutputElement]`: a list of [`VisualQuestionAnsweringOutputElement`] items containing the predicted label and associated probability.

        Raises:
            `InferenceTimeoutError`:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.visual_question_answering(
        ...     image="https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg",
        ...     question="What is the animal doing?"
        ... )
        [
            VisualQuestionAnsweringOutputElement(score=0.778609573841095, answer='laying down'),
            VisualQuestionAnsweringOutputElement(score=0.6957435607910156, answer='sitting'),
        ]
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="visual-question-answering", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=image,
            parameters={"top_k": top_k},
            headers=self.headers,
            model=model_id,
            api_key=self.token,
            extra_payload={"question": question, "image": _b64_encode(image)},
        )
        response = self._inner_post(request_parameters)
        return VisualQuestionAnsweringOutputElement.parse_obj_as_list(response)

    def zero_shot_classification(
        self,
        text: str,
        candidate_labels: List[str],
        *,
        multi_label: Optional[bool] = False,
        hypothesis_template: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[ZeroShotClassificationOutputElement]:
        """
        Provide as input a text and a set of candidate labels to classify the input text.

        Args:
            text (`str`):
                The input text to classify.
            candidate_labels (`List[str]`):
                The set of possible class labels to classify the text into.
            labels (`List[str]`, *optional*):
                (deprecated) List of strings. Each string is the verbalization of a possible label for the input text.
            multi_label (`bool`, *optional*):
                Whether multiple candidate labels can be true. If false, the scores are normalized such that the sum of
                the label likelihoods for each sequence is 1. If true, the labels are considered independent and
                probabilities are normalized for each candidate.
            hypothesis_template (`str`, *optional*):
                The sentence used in conjunction with `candidate_labels` to attempt the text classification by
                replacing the placeholder with the candidate labels.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. This parameter overrides the model defined at the instance level. If not provided, the default recommended zero-shot classification model will be used.


        Returns:
            `List[ZeroShotClassificationOutputElement]`: List of [`ZeroShotClassificationOutputElement`] items containing the predicted labels and their confidence.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example with `multi_label=False`:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> text = (
        ...     "A new model offers an explanation for how the Galilean satellites formed around the solar system's"
        ...     "largest world. Konstantin Batygin did not set out to solve one of the solar system's most puzzling"
        ...     " mysteries when he went for a run up a hill in Nice, France."
        ... )
        >>> labels = ["space & cosmos", "scientific discovery", "microbiology", "robots", "archeology"]
        >>> client.zero_shot_classification(text, labels)
        [
            ZeroShotClassificationOutputElement(label='scientific discovery', score=0.7961668968200684),
            ZeroShotClassificationOutputElement(label='space & cosmos', score=0.18570658564567566),
            ZeroShotClassificationOutputElement(label='microbiology', score=0.00730885099619627),
            ZeroShotClassificationOutputElement(label='archeology', score=0.006258360575884581),
            ZeroShotClassificationOutputElement(label='robots', score=0.004559356719255447),
        ]
        >>> client.zero_shot_classification(text, labels, multi_label=True)
        [
            ZeroShotClassificationOutputElement(label='scientific discovery', score=0.9829297661781311),
            ZeroShotClassificationOutputElement(label='space & cosmos', score=0.755190908908844),
            ZeroShotClassificationOutputElement(label='microbiology', score=0.0005462635890580714),
            ZeroShotClassificationOutputElement(label='archeology', score=0.00047131875180639327),
            ZeroShotClassificationOutputElement(label='robots', score=0.00030448526376858354),
        ]
        ```

        Example with `multi_label=True` and a custom `hypothesis_template`:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()
        >>> client.zero_shot_classification(
        ...    text="I really like our dinner and I'm very happy. I don't like the weather though.",
        ...    labels=["positive", "negative", "pessimistic", "optimistic"],
        ...    multi_label=True,
        ...    hypothesis_template="This text is {} towards the weather"
        ... )
        [
            ZeroShotClassificationOutputElement(label='negative', score=0.9231801629066467),
            ZeroShotClassificationOutputElement(label='pessimistic', score=0.8760990500450134),
            ZeroShotClassificationOutputElement(label='optimistic', score=0.0008674879791215062),
            ZeroShotClassificationOutputElement(label='positive', score=0.0005250611575320363)
        ]
        ```
        """
        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="zero-shot-classification", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=text,
            parameters={
                "candidate_labels": candidate_labels,
                "multi_label": multi_label,
                "hypothesis_template": hypothesis_template,
            },
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        output = _bytes_to_dict(response)
        return [
            ZeroShotClassificationOutputElement.parse_obj_as_instance({"label": label, "score": score})
            for label, score in zip(output["labels"], output["scores"])
        ]

    def zero_shot_image_classification(
        self,
        image: ContentT,
        candidate_labels: List[str],
        *,
        model: Optional[str] = None,
        hypothesis_template: Optional[str] = None,
        # deprecated argument
        labels: List[str] = None,  # type: ignore
    ) -> List[ZeroShotImageClassificationOutputElement]:
        """
        Provide input image and text labels to predict text labels for the image.

        Args:
            image (`Union[str, Path, bytes, BinaryIO, PIL.Image.Image]`):
                The input image to caption. It can be raw bytes, an image file, a URL to an online image, or a PIL Image.
            candidate_labels (`List[str]`):
                The candidate labels for this image
            labels (`List[str]`, *optional*):
                (deprecated) List of string possible labels. There must be at least 2 labels.
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. This parameter overrides the model defined at the instance level. If not provided, the default recommended zero-shot image classification model will be used.
            hypothesis_template (`str`, *optional*):
                The sentence used in conjunction with `candidate_labels` to attempt the image classification by
                replacing the placeholder with the candidate labels.

        Returns:
            `List[ZeroShotImageClassificationOutputElement]`: List of [`ZeroShotImageClassificationOutputElement`] items containing the predicted labels and their confidence.

        Raises:
            [`InferenceTimeoutError`]:
                If the model is unavailable or the request times out.
            `HTTPError`:
                If the request fails with an HTTP error status code other than HTTP 503.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient()

        >>> client.zero_shot_image_classification(
        ...     "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg",
        ...     labels=["dog", "cat", "horse"],
        ... )
        [ZeroShotImageClassificationOutputElement(label='dog', score=0.956),...]
        ```
        """
        # Raise ValueError if input is less than 2 labels
        if len(candidate_labels) < 2:
            raise ValueError("You must specify at least 2 classes to compare.")

        model_id = model or self.model
        provider_helper = get_provider_helper(self.provider, task="zero-shot-image-classification", model=model_id)
        request_parameters = provider_helper.prepare_request(
            inputs=image,
            parameters={
                "candidate_labels": candidate_labels,
                "hypothesis_template": hypothesis_template,
            },
            headers=self.headers,
            model=model_id,
            api_key=self.token,
        )
        response = self._inner_post(request_parameters)
        return ZeroShotImageClassificationOutputElement.parse_obj_as_list(response)

    def get_endpoint_info(self, *, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about the deployed endpoint.

        This endpoint is only available on endpoints powered by Text-Generation-Inference (TGI) or Text-Embedding-Inference (TEI).
        Endpoints powered by `transformers` return an empty payload.

        Args:
            model (`str`, *optional*):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. This parameter overrides the model defined at the instance level. Defaults to None.

        Returns:
            `Dict[str, Any]`: Information about the endpoint.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient("meta-llama/Meta-Llama-3-70B-Instruct")
        >>> client.get_endpoint_info()
        {
            'model_id': 'meta-llama/Meta-Llama-3-70B-Instruct',
            'model_sha': None,
            'model_dtype': 'torch.float16',
            'model_device_type': 'cuda',
            'model_pipeline_tag': None,
            'max_concurrent_requests': 128,
            'max_best_of': 2,
            'max_stop_sequences': 4,
            'max_input_length': 8191,
            'max_total_tokens': 8192,
            'waiting_served_ratio': 0.3,
            'max_batch_total_tokens': 1259392,
            'max_waiting_tokens': 20,
            'max_batch_size': None,
            'validation_workers': 32,
            'max_client_batch_size': 4,
            'version': '2.0.2',
            'sha': 'dccab72549635c7eb5ddb17f43f0b7cdff07c214',
            'docker_label': 'sha-dccab72'
        }
        ```
        """
        if self.provider != "hf-inference":
            raise ValueError(f"Getting endpoint info is not supported on '{self.provider}'.")

        model = model or self.model
        if model is None:
            raise ValueError("Model id not provided.")
        if model.startswith(("http://", "https://")):
            url = model.rstrip("/") + "/info"
        else:
            url = f"{constants.INFERENCE_ENDPOINT}/models/{model}/info"

        response = get_session().get(url, headers=build_hf_headers(token=self.token))
        hf_raise_for_status(response)
        return response.json()

    def health_check(self, model: Optional[str] = None) -> bool:
        """
        Check the health of the deployed endpoint.

        Health check is only available with Inference Endpoints powered by Text-Generation-Inference (TGI) or Text-Embedding-Inference (TEI).

        Args:
            model (`str`, *optional*):
                URL of the Inference Endpoint. This parameter overrides the model defined at the instance level. Defaults to None.

        Returns:
            `bool`: True if everything is working fine.

        Example:
        ```py
        >>> from huggingface_hub import InferenceClient
        >>> client = InferenceClient("https://jzgu0buei5.us-east-1.aws.endpoints.huggingface.cloud")
        >>> client.health_check()
        True
        ```
        """
        if self.provider != "hf-inference":
            raise ValueError(f"Health check is not supported on '{self.provider}'.")

        model = model or self.model
        if model is None:
            raise ValueError("Model id not provided.")
        if not model.startswith(("http://", "https://")):
            raise ValueError("Model must be an Inference Endpoint URL.")
        url = model.rstrip("/") + "/health"

        response = get_session().get(url, headers=build_hf_headers(token=self.token))
        return response.status_code == 200

    @property
    def chat(self) -> "ProxyClientChat":
        return ProxyClientChat(self)


class _ProxyClient:
    """Proxy class to be able to call `client.chat.completion.create(...)` as OpenAI client."""

    def __init__(self, client: InferenceClient):
        self._client = client


class ProxyClientChat(_ProxyClient):
    """Proxy class to be able to call `client.chat.completion.create(...)` as OpenAI client."""

    @property
    def completions(self) -> "ProxyClientChatCompletions":
        return ProxyClientChatCompletions(self._client)


class ProxyClientChatCompletions(_ProxyClient):
    """Proxy class to be able to call `client.chat.completion.create(...)` as OpenAI client."""

    @property
    def create(self):
        return self._client.chat_completion
