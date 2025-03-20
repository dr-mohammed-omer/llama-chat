import Replicate from "replicate";
import { ReplicateStream, StreamingTextResponse } from "ai";

export const runtime = "edge";

const VERSIONS = {
  "yorickvp/llava-13b":
    "e272157381e2a3bf12df3a8edd1f38d1dbd736bbb7437277c8b34175f8fce358",
  "nateraw/salmonn":
    "ad1d3f9d2bd683628242b68d890bef7f7bd97f738a7c2ccbf1743a594c723d83",
};

// ✅ Handle GET requests (prevents 405 errors)
export async function GET() {
  return new Response("GET method not supported for this endpoint", {
    status: 405,
  });
}

// ✅ Handle POST requests
export async function POST(req) {
  try {
    const params = await req.json();
    const ip =
      req.headers.get("x-real-ip") || req.headers.get("x-forwarded-for");

    if (!params.replicateApiToken) {
      return new Response("Missing API token", { status: 400 });
    }

    params.replicateClient = new Replicate({
      auth: params.replicateApiToken,
      userAgent: "llama-chat",
    });

    if (!ip) {
      console.error("IP address is null");
      return new Response("IP address could not be retrieved", { status: 500 });
    }

    let response;
    if (params.image) {
      response = await runLlava(params);
    } else if (params.audio) {
      response = await runSalmonn(params);
    } else {
      response = await runLlama(params);
    }

    // Convert the response into a streaming text response
    const stream = await ReplicateStream(response);
    return new StreamingTextResponse(stream);
  } catch (error) {
    console.error("Error in POST request:", error);
    return new Response("Internal Server Error", { status: 500 });
  }
}

// ✅ Function to run Llama model
async function runLlama({
  replicateClient,
  model,
  prompt,
  systemPrompt,
  maxTokens,
  temperature,
  topP,
}) {
  console.log("Running Llama model:", model);

  return await replicateClient.predictions.create({
    model,
    stream: true,
    input: {
      prompt,
      max_new_tokens: maxTokens,
      ...(model.includes("llama3")
        ? { max_tokens: maxTokens }
        : { max_new_tokens: maxTokens }),
      temperature,
      repetition_penalty: 1,
      top_p: topP,
    },
  });
}

// ✅ Function to run Llava model (for image-based AI)
async function runLlava({
  replicateClient,
  prompt,
  maxTokens,
  temperature,
  topP,
  image,
}) {
  console.log("Running Llava model");

  return await replicateClient.predictions.create({
    stream: true,
    input: {
      prompt,
      top_p: topP,
      temperature,
      max_tokens: maxTokens,
      image,
    },
    version: VERSIONS["yorickvp/llava-13b"],
  });
}

async function runSalmonn({
  replicateClient,
  prompt,
  maxTokens,
  temperature,
  topP,
  audio,
}) {
  console.log("Running Salmonn model");

  return await replicateClient.predictions.create({
    stream: true,
    input: {
      prompt,
      top_p: topP,
      temperature,
      max_length: maxTokens,
      wav_path: audio,
    },
    version: VERSIONS["nateraw/salmonn"],
  });
}
