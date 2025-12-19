# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Activate your conda environment (uncomment if needed)
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate /data/user_data/ssaxena2/tdmpc2-jax
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ogpo

# Set up variables
log=True
seed=82723
env_name=wx200-square
run_group="wx200_square"
run_name="wx200_square"
wandb_name="wx200_square"
horizon_length=4
bc_coeff=0.1
utd_warmup=1
utd_online=1
tau=0.05
discount=0.991
SAVE_DIR="/data/user_data/$USER/ogpo"
restore_actor_path=None
restore_critic_path=None
ep_resume=0
offline_steps=1000000
calql_steps=0
q_warmup_steps=0
online_steps=0
best_of_n=8
num_qs=10
q_agg=mean
subsample_bon=True
offline_ratio=0.0
clip_bc=True
use_success_buffer=True
plot_q_vs_mc=True
clip_min_epsilon_multiplier=1.0

for arg in "$@"; do
  case $arg in
    --*=*)
      key="${arg%%=*}"      # part before '='
      value="${arg#*=}"     # part after '='
      key="${key#--}"       # strip leading '--'
      eval "$key=\"$value\"" # set variable dynamically
      ;;
    *)
      echo "Unknown argument: $arg"
      ;;
  esac
done

echo "Starting QPPO training job on $(hostname) at $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
if command -v nvidia-smi &> /dev/null; then
    echo "Using GPU: $(nvidia-smi -L)"
fi

cd ../..

# Run the training
for seed in $seed
do
    echo "Running with seed: $seed"
    python main_ogpo_real.py \
    --agent=agents/reinflow_calql.py \
    --project=OGPO_hardware \
    --run_group=$run_group \
    --run_name=$run_name \
    --wandb_name=$wandb_name \
    --log=$log \
    --restore_actor_path=$restore_actor_path \
    --restore_critic_path=$restore_critic_path \
    --ep_resume=$ep_resume \
    --offline_steps=$offline_steps \
    --calql_steps=$calql_steps \
    --q_warmup_steps=$q_warmup_steps \
    --online_steps=$online_steps \
    --clip_bc=$clip_bc \
    --use_success_buffer=$use_success_buffer \
    --plot_q_vs_mc=$plot_q_vs_mc \
    --best_of_n=$best_of_n \
    --agent.q_agg=$q_agg \
    --agent.subsample_bon=$subsample_bon \
    --agent.num_qs=$num_qs \
    --seed=$seed \
    --discount=$discount \
    --agent.tau=$tau \
    --utd_warmup=$utd_warmup \
    --utd_online=$utd_online \
    --env_name=$env_name \
    --sparse=False \
    --horizon_length=$horizon_length \
    --agent.flow_steps=10 \
    --n_eval_envs=1 \
    --eval_episodes=3 \
    --log_interval=5000 \
    --eval_interval=20000 \
    --eval_interval_bc=300000 \
    --save_interval=300000 \
    --save_dir=$SAVE_DIR \
    --agent.clip_epsilon=0.01 \
    --agent.entropy_coeff=0.0 \
    --agent.min_noise_std=0.005 \
    --agent.max_noise_std=0.005 \
    --agent.use_constant_noise=True \
    --agent.constant_noise_std=0.005 \
    --agent.ppo_batch_size=256 \
    --agent.grpo_num_samples=32 \
    --offline_ratio=$offline_ratio \
    --agent.use_bc_regularization=True \
    --agent.bc_coeff=$bc_coeff \
    --start_training=10000 \
    --agent.actor_scheduler=cosine \
    --agent.critic_scheduler=constant \
    --agent.policy_type=flow \
    --agent.clip_min_epsilon_multiplier=$clip_min_epsilon_multiplier
done

echo "Job completed at $(date)"

