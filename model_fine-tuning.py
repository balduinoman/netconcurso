from unsloth import FastLanguageModel
from transformers import TextStreamer
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import torch


def gerar_prompt_sql(contexto, pergunta, resposta=''):
  return f'''Você é um modelo poderoso de texto-para-SQL. Seu trabalhoé responder perguntas sobre um banco de dados. Você recebe uma pergunta e um contexto relacionado a uma ou mais tabelas.

  Você deve gerar a consulta SQL que responde a pergunta.

  ### Instruction:
  Contexto: {contexto}

  ### Input:
  Pergunta: {pergunta}

  ### Response:
  Resposta: {resposta}
  '''

def formatar_prompts(dados):
  contextos = dados['contexto']
  perguntas = dados['pergunta']
  respostas = dados['resposta']
  textos = []
  for contexto, pergunta, resposta in zip(contextos, perguntas, respostas):
    texto = gerar_prompt_sql(contexto, pergunta, resposta) + EOS_TOKEN
    textos.append(texto)
  return {'texto': textos}

checkpoint_modelo = 'unsloth/Meta-Llama-3.1-8B'

modelo, tokenizador = FastLanguageModel.from_pretrained(
    model_name = checkpoint_modelo,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True
)

EOS_TOKEN = tokenizador.eos_token

modelo = FastLanguageModel.get_peft_model(
    modelo,
    r = 16,
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = 'none',
    use_gradient_checkpointing = 'unsloth',
    random_state = 10,
    use_rslora = False,
    loftq_config = None
)

dataset = load_dataset('emdemor/sql-create-context-pt', split = 'train')

trainer = SFTTrainer(
    model = modelo,
    tokenizer = tokenizador,
    train_dataset = dataset,
    dataset_text_field = 'texto',
    max_seq_length = 2048,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        learning_rate = 2e-5,
        max_steps = 60,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = 'adamw_8bit',
        weight_decay = 0.01,
        lr_scheduler_type = 'linear',
        seed = 10,
        output_dir = 'output'
    )
)

FastLanguageModel.for_inference(modelo)

prompt_sql = gerar_prompt_sql(
        'CREATE TABLE head (age INTEGER)',
        'Quantas pessoas tem mais de 56 anos?',
        ''
    )

prompt_tokenizado = tokenizador([prompt_sql], return_tensors='pt').to('cuda')

streamer_texto = TextStreamer(tokenizador)

resposta = modelo.generate(**prompt_tokenizado, streamer = streamer_texto, max_new_tokens=64)

modelo.save_pretrained_gguf('modelo', tokenizador, quantization_method='q4_k_m')