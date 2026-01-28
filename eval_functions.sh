#!/bin/bash

# Wav2Vec Unsupervised Pipeline Runner
# This script runs the entire fairseq wav2vec unsupervised pipeline
# with checkpointing to allow resuming from any step

set -e  # Exit on error
set -o pipefail  # Exit if any command in a pipe fails

# ==================== CONFIGURATION ====================
# Set these variables according to your environment and needs

# Main directories
#.... directories to add to root.......
DIR_PATH="$HOME/unsupervised_wav"
DATA_ROOT="$DIR_PATH/data" 
FAIRSEQ_ROOT="$DIR_PATH/fairseq"
KENLM_ROOT="$DIR_PATH/kenlm/build/bin"  # Path to KenLM installation
VENV_PATH="$DIR_PATH/venv"    # Path to virtual environment 
KALDI_ROOT="$DIR_PATH/pykaldi/tools/kaldi"

# Dataset specifics
DATASET="librispeech"

# Output directories (will be created if they don't exist)
CLUSTERING_DIR="$DATA_ROOT/clustering/$DATASET"
LM_DIR="$DATA_ROOT/language_models/$DATASET"
RESULTS_DIR=$1
CHECKPOINT_DIR="$DATA_ROOT/checkpoints/$DATASET_NAME"
LOG_DIR="$DATA_ROOT/logs/$DATASET"
TEXT_OUTPUT="$DATA_ROOT/text"
GANS_OUTPUT_PHONES="$DATA_ROOT/transcription_phones"
GANS_OUTPUT_WORDS="$DATA_ROOT/transcription_words"
ST_OUTPUT="$DATA_ROOT/selftraining"

# Checkpoint file to track progress
CHECKPOINT_FILE="$CHECKPOINT_DIR/progress.checkpoint"

# ==================== HELPER FUNCTIONS ====================

# Create directories if they don't exist

create_dirs() {
    mkdir -p  "$GANS_OUTPUT_PHONES" "$GANS_OUTPUT_PHONES" "$ST_OUTPUT"
}

# Log message with timestamp
log() {
    local message="$1"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] $message" | tee -a "$LOG_DIR/pipeline.log"
}

# Check if a step has been completed
is_completed() {
    local step="$1"
    if [ -f "$CHECKPOINT_FILE" ]; then
        grep -q "^$step:COMPLETED$" "$CHECKPOINT_FILE" && return 0
    fi
    return 1
}

# Mark a step as completed
mark_completed() {
    local step="$1"
    echo "$step:COMPLETED" >> "$CHECKPOINT_FILE"
    log "Marked step '$step' as completed"
}

# Mark a step as in progress
mark_in_progress() {
    local step="$1"
    # First remove any existing in-progress markers for this step
    if [ -f "$CHECKPOINT_FILE" ]; then
        sed -i "/^$step:IN_PROGRESS$/d" "$CHECKPOINT_FILE"
    fi
    echo "$step:IN_PROGRESS" >> "$CHECKPOINT_FILE"
    log "Marked step '$step' as in progress"
}

# Check if a step is in progress (for recovery after crash)
is_in_progress() {
    local step="$1"
    if [ -f "$CHECKPOINT_FILE" ]; then
        grep -q "^$step:IN_PROGRESS$" "$CHECKPOINT_FILE" && return 0
    fi
    return 1
}

# Activate virtual environment if provided
activate_venv() {
    if [ -n "$VENV_PATH" ]; then
        log "Activating virtual environment at $VENV_PATH"
        source "$VENV_PATH/bin/activate"
    fi
}

setup_path() {

    export HYDRA_FULL_ERROR=1
    export LD_LIBRARY_PATH="${KALDI_ROOT}/src/lib:${KENLM_ROOT}/lib:${LD_LIBRARY_PATH:-}"

}


#script is used to update a yaml file 
update_yaml_config() {
    # Check for minimum number of arguments: config file and at least one update.
    if [ "$#" -lt 2 ]; then
        echo "Usage: update_yaml_config <config_file> <key=value> [<key=value> ...]"
        echo "Example: update_yaml_config config.yaml 'user.name=Alice' 'db.timeout=60'"
        return 1
    fi

    local CONFIG_FILE="$1"
    shift  # Remove config file from arguments
    
    # Check if the config file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: File '$CONFIG_FILE' does not exist." >&2
        return 1
    fi

    local YQ_COMMAND=""
    local updates_applied=""

    # 1. Loop through all key=value arguments
    for ARG in "$@"; do
        # Split ARG into KEY and VALUE at the first '='
        KEY="${ARG%%=*}"
        VALUE="${ARG#*=}"

        # 2. Build the yq assignment string
        # yq requires the path to start with '.' and the value to be quoted
        # to ensure it's treated as a string, not a variable or path.
        # Format: '.key.path = "value"'
        # We use single quotes around the full filter to protect the double quotes for the value.
        YQ_COMMAND="$YQ_COMMAND .$KEY = \"$VALUE\" |"
        
        updates_applied="$updates_applied$KEY = $VALUE, "
    done

    # 3. Remove the trailing '|' (pipe symbol) from the command string
    # This prevents an invalid yq filter.
    YQ_COMMAND="${YQ_COMMAND% |}"

    # --- Apply Update ---
    
    echo "Updating file: $CONFIG_FILE"
    echo "Applying updates: ${updates_applied%, }"
    
    # Execute yq:
    # -i: Edit file in-place
    # -y: Ensure output remains YAML (important for file structure)
    yq -i -y "$YQ_COMMAND" "$CONFIG_FILE"

    if [ $? -eq 0 ]; then
        echo "Configuration file '$CONFIG_FILE' updated successfully."
        return 0
    else
        echo "Error: Failed to update '$CONFIG_FILE'. Check yq version and command." >&2
        return 1
    fi
}

# Function to update a file's empty variables with provided values
update_file_variables() {
    if [ "$#" -lt 2 ]; then
        echo "Usage: update_file_variables <filename> <var1=value> <var2=value> ..."
        return 1
    fi

    local file="$1"
    shift  # Remove filename from arguments

    # Create a backup of the original file
    # cp "$file" "$file.bak"

    # Loop through remaining arguments (variable=value pairs)
    for arg in "$@"; do
        local var_name=$(echo "$arg" | cut -d '=' -f 1)   # Extract variable name
        local var_value=$(echo "$arg" | cut -d '=' -f 2-) # Extract everything after '='

        # Use `sed` to replace only existing variables in the file
        echo $var_name
        echo $var_value
        local escaped_value=$(echo "$var_value" | sed 's/[\/&]/\\&/g')

        # Use `sed` to replace only existing variables in the file
        sed -i "s|^$var_name=.*$|$var_name=$escaped_value|" "$file"
        # sed -i "s/^$var_name=.*$/$var_name=$var_value/" "$file"
        echo "done"
    done

    echo "Updated $file successfully."
}


comment_line() {
    # Ensure correct number of arguments
    if [ "$#" -ne 2 ]; then
        echo "Usage: comment_line <file_name> <line_to_comment>"
        return 1
    fi

    file="$1"         # File to modify
    line_to_find="$2" # Line of code to comment out

    # Escape special characters in the input to prevent sed errors
    escaped_line_to_find=$(printf "%s" "$line_to_find" | sed 's/[\/&]/\\&/g')

    # Check if the line exists in the file
    if grep -Fxq "$line_to_find" "$file"; then
        echo "Found line in $file: $line_to_find"

        # Backup the file before modification
        cp "$file" "$file.bak"

        # Comment out the matching line
        sed -i "s|^$escaped_line_to_find|# $escaped_line_to_find|" "$file"

        echo "Commented out the line in $file"
    else
        echo "Line not found in $file. No changes made."
    fi
}



get_best_path_pipeline() {
    local input_file="$1"

    # --- Input Validation ---
    if [ -z "$input_file" ]; then
        echo "Usage: get_best_path_pipeline <input_log_file>" >&2
        return 1
    fi
    if [ ! -r "$input_file" ]; then
        echo "Error: File '$input_file' not found or not readable." >&2
        return 1
    fi
    # --- End Input Validation ---

    # 1. Filter lines containing both "INFO:root:" and "wer [...]%"
    #    Use grep -E for extended regex OR chain two greps. Chaining is safer.
    local filtered_lines
    filtered_lines=$(grep 'INFO:root:' "$input_file" | grep 'wer [0-9.]*%' || true)
    # '|| true' prevents script exit if grep finds nothing

    # Check if any lines matched
    if [ -z "$filtered_lines" ]; then
         # echo "Warning: No lines matching required format found." >&2 # Optional warning
         return 0 # Indicate success, but no result found
    fi

    # 2. Transform matching lines to "WER PATH" format using POSIX awk
    #    This awk script extracts WER and path from the already filtered lines.
    local transformed_lines
    transformed_lines=$(echo "$filtered_lines" | awk '
    {
        # Find WER using match() and substr()
        if (match($0, /wer [0-9.]+\%/)) {
             wer_match_str = substr($0, RSTART, RLENGTH); # e.g., "wer 71.71%"
             if (match(wer_match_str, /[0-9.]+/)) {
                 current_wer = substr(wer_match_str, RSTART, RLENGTH);
             } else { next; } # Skip if number extraction fails
        } else { next; } # Skip if WER pattern not found

        # Find Path using match() and substr()
        if (match($0, /INFO:root:[^:]+:/)) {
            path_match_str = substr($0, RSTART, RLENGTH); # e.g., "INFO:root:/path:"
            sub(/^INFO:root:/, "", path_match_str);
            sub(/:$/, "", path_match_str);
            current_path = path_match_str;
        } else { next; } # Skip if path pattern not found

        # Print in "WER PATH" format
        print current_wer, current_path;
    }')

    if [ -z "$transformed_lines" ]; then
        # This might happen if extraction failed even after initial grep
        # echo "Warning: Could not extract WER/Path from matching lines." >&2
        return 0
    fi

    # 3. Sort numerically based on the first field (WER)
    # 4. Take the first line (lowest WER)
    local best_line
    best_line=$(echo "$transformed_lines" | sort -n -k1,1 | head -n 1)

    # 5. Extract the path (everything after the first space)
    #    Using awk is robust for paths with spaces
    local best_path
    best_path=$(echo "$best_line" | awk '{ $1=""; sub(/^ /,""); print }')

    echo "$best_path"
    return 0
}



update_script_with_condition() {
    local SCRIPT_FILE="$1"
    
    # Create a backup of the original file
    cp "$SCRIPT_FILE" "$SCRIPT_FILE.bak"
    echo "Backup created at ${SCRIPT_FILE}.bak"
    
    # Export the filename so the Python code can access it
    export SCRIPT_FILE
    
    python3 << 'EOF'
import re
import sys
import os

    # Retrieve the target script filename from the environment variable
script_file = os.environ['SCRIPT_FILE']

    # Define a regex pattern that matches the ground truth cp command
    # The pattern is designed to match:
    # cp $data_dir/$x/{feats.scp,cmvn.scp,utt2spk,spk2utt} $data_dir/$x_gt/
pattern = re.compile(r'cp\s+\$data_dir/\$x/\{feats\.scp,cmvn\.scp,utt2spk,spk2utt\}\s+\$data_dir/\$x_gt/')

    # The conditional block to insert
conditional_block = '''if [[ "$x" == "$valid_name" ]]; then
        python local/copy_aligned_text.py < $w2v_dir/$x.$label > $data_dir/$x_gt/text
fi
    '''

    # Read the original script
with open(script_file, 'r') as f:
        lines = f.readlines()

new_lines = []
found = False

for line in lines:
        new_lines.append(line)
        # When the line matches our pattern, insert the conditional block immediately after it
        if pattern.search(line):
            found = True
            # Prevent duplicate insertion by checking if the conditional block already exists
            if not any('if [[ "$x" == "$valid_name" ]]; then' in l for l in lines):
                new_lines.append(conditional_block + "\n")

if not found:
        sys.stderr.write("Pattern not found in file. No modifications done.\n")
        sys.exit(1)

    # Write back the modified script
with open(script_file, 'w') as f:
        f.writelines(new_lines)

print(f"Modification completed. {script_file} updated.")
EOF
}



#============================ model evaLuation =======================

transcription_gans_viterbi(){

   export HYDRA_FULL_ERROR=1
   export FAIRSEQ_ROOT=$FAIRSEQ_ROOT
   export KALDI_ROOT="$DIR_PATH/pykaldi/tools/kaldi"
   export KENLM_ROOT="$KENLM_ROOT"
   export PYTHONPATH=$FAIRSEQ_ROOT:$PYTHONPATH
#    

#updating parameters viterbi.yaml 
update_yaml_config "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/generate/viterbi.yaml" fairseq.task.data="$CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled" fairseq.task.text_data="$TEXT_OUTPUT/phones/" fairseq.common_eval.path="$RESULTS_DIR/checkpoint_best.pt" fairseq.dataset.batch_size=1 fairseq.dataset.num_workers=0 fairseq.dataset.required_batch_size_multiple=1 fairseq.dataset.gen_subset=valid results_path="$GANS_OUTPUT_PHONES"

#evaluating the GANS models for validation phones
python "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/w2vu_generate.py" --config-dir "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/generate" \
 --config-name viterbi fairseq.common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" \
  fairseq.task.data="$CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled" \
  fairseq.common_eval.path="$RESULTS_DIR/checkpoint_best.pt" \
  fairseq.dataset.gen_subset=valid results_path="$GANS_OUTPUT_PHONES"


#evaluating the GANS models for validation phones
  python "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/w2vu_generate.py" --config-dir "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/generate" \
 --config-name viterbi fairseq.common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" \
  fairseq.task.data="$CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled" \
  fairseq.common_eval.path="$RESULTS_DIR/checkpoint_best.pt" \
  fairseq.dataset.gen_subset=train results_path="$GANS_OUTPUT_PHONES"

}

transcription_gans_kaldi(){
   activate_venv 

   # first step is to make a copy of viterbi and name is as kaldi
   cp -r $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/generate/viterbi.yaml $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/generate/kaldi.yaml

 
   export HYDRA_FULL_ERROR=1
   export FAIRSEQ_ROOT=$FAIRSEQ_ROOT
   export KALDI_ROOT="$DIR_PATH/pykaldi/tools/kaldi"
   export KENLM_ROOT="$KENLM_ROOT"
   export PYTHONPATH=$FAIRSEQ_ROOT:$PYTHONPATH
  
update_yaml_config "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/generate/kaldi.yaml" fairseq.common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" fairseq.task.data="$CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled" fairseq.common_eval.path="$RESULTS_DIR/checkpoint_best.pt" kaldi_decoder_config.hlg_graph_path="$TEXT_OUTPUT/fst/phn_to_words_sil/HLGa.phn.kenlm.wrd.o40003.fst" kaldi_decoder_config.output_dict=$TEXT_OUTPUT/fst/phn_to_words_sil/kaldi_dict.kenlm.wrd.o40003.txt fairseq.task.labels="wrd" w2l_decoder="KALDI" fairseq.dataset.gen_subset=train fairseq.dataset.batch_size=1 fairseq.dataset.num_workers=0 fairseq.dataset.required_batch_size_multiple=1 results_path="$GANS_OUTPUT_WORDS" 

#evaluating for train words
python "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/w2vu_generate.py" --config-dir "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/generate" \
 --config-name kaldi fairseq.common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" \
  fairseq.task.data="$CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled" \
  fairseq.common_eval.path="$RESULTS_DIR/checkpoint_best.pt" \
  kaldi_decoder_config.hlg_graph_path="$TEXT_OUTPUT/fst/phn_to_words_sil/HLGa.phn.kenlm.wrd.o40003.fst" \
  kaldi_decoder_config.output_dict=$TEXT_OUTPUT/fst/phn_to_words_sil/kaldi_dict.kenlm.wrd.o40003.txt \
  fairseq.task.labels="wrd" \
   w2l_decoder="KALDI" \
  fairseq.dataset.gen_subset=train results_path="$GANS_OUTPUT_WORDS" 

 #evaluating for validation words
  python "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/w2vu_generate.py" --config-dir "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/generate" \
 --config-name kaldi fairseq.common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" \
  fairseq.task.data="$CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled" \
  fairseq.common_eval.path="$RESULTS_DIR/checkpoint_best.pt" \
  kaldi_decoder_config.hlg_graph_path="$TEXT_OUTPUT/fst/phn_to_words_sil/HLGa.phn.kenlm.wrd.o40003.fst" \
  kaldi_decoder_config.output_dict=$TEXT_OUTPUT/fst/phn_to_words_sil/kaldi_dict.kenlm.wrd.o40003.txt \
  fairseq.task.labels="wrd" \
   w2l_decoder="KALDI" \
  fairseq.dataset.gen_subset=valid results_path="$GANS_OUTPUT_WORDS" 
  

}




#======================self training -====================

self_training()
{
   export HYDRA_FULL_ERROR=1
   export FAIRSEQ_ROOT=$FAIRSEQ_ROOT
   export KALDI_ROOT="$DIR_PATH/pykaldi/tools/kaldi"
   export KENLM_ROOT="$KENLM_ROOT"
   export PYTHONPATH=$FAIRSEQ_ROOT:$PYTHONPATH

   
   setup_path
   
   #very important step  copy 
   cp -r $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/kaldi_self_train $KALDI_ROOT/egs 

    TRAIN_FILE=$KALDI_ROOT/egs/kaldi_self_train/st/train.sh 

    update_file_variables $TRAIN_FILE w2v_dir="$CLUSTERING_DIR" lab_dir=$GANS_OUTPUT_PHONES out_dir=$ST_OUTPUT arpa_lm="$TEXT_OUTPUT/phones/lm.phones.filtered.04.arpa" arpa_lm_bin="$TEXT_OUTPUT/phones/lm.phones.filtered.04.bin/" label=phnc

    comment_line $TRAIN_FILE "  python local/copy_aligned_text.py < \$w2v_dir/\$x.\$label > \$data_dir/\$x_gt/text"

    update_script_with_condition $TRAIN_FILE

    cd $KALDI_ROOT/egs/kaldi_self_train/st/ 
    chmod +x $TRAIN_FILE
    $TRAIN_FILE > $KALDI_ROOT/egs/kaldi_self_train/st/results.txt

}



transcription_HMM_phone_eval()
{
    export FAIRSEQ_ROOT=$FAIRSEQ_ROOT
   export KALDI_ROOT="$DIR_PATH/pykaldi/tools/kaldi"
   export KENLM_ROOT="$KENLM_ROOT"
   export PYTHONPATH=$FAIRSEQ_ROOT:$PYTHONPATH

   
 DECODE_PHONE=$KALDI_ROOT/egs/kaldi_self_train/st/decode_phone.sh

 output=$(get_best_path_pipeline $KALDI_ROOT/egs/kaldi_self_train/st/results.txt) #created an output to store best hmm results 
 IFS='/' read -ra ADDR <<< "$output"
 result="${ADDR[-1]%.tra.txt}"

update_file_variables $DECODE_PHONE out_dir=$ST_OUTPUT dec_lmparam="${ADDR[-1]%.tra.txt}" dec_exp=${ADDR[-4]} dec_script=$KALDI_ROOT/egs/kaldi_self_train/st/decode.sh dec_splits="valid"
chmod +x  $DECODE_PHONE
cd $KALDI_ROOT/egs/kaldi_self_train/st/ 

$DECODE_PHONE
}

transcription_HMM_word_eval()
{
    export FAIRSEQ_ROOT=$FAIRSEQ_ROOT
   export KALDI_ROOT="$DIR_PATH/pykaldi/tools/kaldi"
   export KENLM_ROOT="$KENLM_ROOT"
   export PYTHONPATH=$FAIRSEQ_ROOT:$PYTHONPATH
 
 DECODE_WORD=$KALDI_ROOT/egs/kaldi_self_train/st/decode_word_step1.sh

 output=$(get_best_path_pipeline $KALDI_ROOT/egs/kaldi_self_train/st/results.txt) #results.txt is an output that stores best hmm results 
 IFS='/' read -ra ADDR <<< "$output"
 result="${ADDR[-1]%.tra.txt}"

update_file_variables $DECODE_WORD w2v_dir="$CLUSTERING_DIR" out_dir=$ST_OUTPUT lexicon=$TEXT_OUTPUT/lexicon_filtered.lst wrd_arpa_lm=$TEXT_OUTPUT/kenlm.wrd.o40003.arpa wrd_arpa_lm_bin=$TEXT_OUTPUT/kenlm.wrd.o40003.bin dec_exp=${ADDR[-4]} dec_splits="valid" dec_script=steps/decode_fmllr.sh

chmod +x  $DECODE_WORD

cd $KALDI_ROOT/egs/kaldi_self_train/st/ 
$DECODE_WORD > $KALDI_ROOT/egs/kaldi_self_train/st/results_word.txt

}

transcription_HMM_word2_eval()
{
   export FAIRSEQ_ROOT=$FAIRSEQ_ROOT
   export KALDI_ROOT="$DIR_PATH/pykaldi/tools/kaldi"
   export KENLM_ROOT="$KENLM_ROOT"
   export PYTHONPATH=$FAIRSEQ_ROOT:$PYTHONPATH
   
 DECODE_WORD2=$KALDI_ROOT/egs/kaldi_self_train/st/decode_word_step2.sh

 output=$(get_best_path_pipeline $KALDI_ROOT/egs/kaldi_self_train/st/results_word.txt) #created an output to store best hmm results  #not necessary, we can view and place our values 
 IFS='/' read -ra ADDR <<< "$output"
 result="${ADDR[-1]%.tra.txt}"

dec_splits="train valid"
update_file_variables $DECODE_WORD2 out_dir=$ST_OUTPUT dec_exp=${ADDR[-4]} dec_splits="valid" dec_lmparam="${ADDR[-1]%.tra.txt}"
sed -i 's|\(decode\${dec_suffix}_[^/]*\)/scoring|\1.si/scoring|g' "$DECODE_WORD2"

chmod +x  $DECODE_WORD2
echo "here"
cd $KALDI_ROOT/egs/kaldi_self_train/st/ 
$DECODE_WORD2

}



main(){

   create_dirs
   activate_venv 
#the trained checkpoints from train_gans will be stored in a folder called multirun. The checkpoint will be stored in this format 
#multirun --
 #         |
 #         day/month/year --
 #                         |
 #                         time --
 #                                |
 #                                checkpoint_best.pt
 #                                 checkpoint_last.pt
 #therefore it is advisable to manually provide the path to the exact checkpoint to use under the variable $CHECKPOINT_DIR  in the run_wav2vec.sh script
 
'''
Transcriptions from the GAN model 
     transcription_gans_viterbi: outputs phonetic transcription in variable name $GANS_OUTPUT_PHONES
     transcription_gans_kaldi: outputs word transcription in variable name $GANS_OUTPUT_WORDS
'''
    transcription_gans_viterbi  #for these we need both train and validation since the train will be used by the HMM
    transcription_gans_kaldi

'''
  Does Hidden Markov training on the outputs from the transcription_gans_viterbi and prepare_audio 
 outputs three HMM 
'''
    self_training

'''
transcribes and evaluates the validation set using the best HMM Model 
'''
    transcription_HMM_phone_eval
    transcription_HMM_word_eval
    transcription_HMM_word2_eval
    
    log "Pipeline completed successfully!"
}

main
