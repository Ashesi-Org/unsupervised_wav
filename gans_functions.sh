# this is to help updates yaml files needed for the successful run of certain processes.
add_to_existing_yaml() {
  if [ "$#" -ne 4 ]; then
    echo "Usage: add_to_existing_yaml <config_file> <parent_key> <new_key> <new_value>"
    return 1
  fi

  local config_file="$1"
  local parent_key="$2"
  local new_key="$3"
  local new_value="$4"
  local temp_file="${config_file}.tmp"

  if [ ! -f "$config_file" ]; then
    echo "Error: File '$config_file' not found."
    return 1
  fi

  # Convert input to a valid yq-compatible path
  local full_path=".$parent_key.$new_key"


  # Check if the parent key exists
#  
    yq -y  "$full_path = \"$new_value\"" "$config_file" > "$temp_file"

    echo "Updated '$parent_key' with '$new_key: $new_value'."


  if [ $? -eq 0 ]; then

    mv "$temp_file" "$config_file"
    # We no longer need to check if the parent exists, yq handles creation automatically.
    echo "Configuration file '$config_file' updated successfully."
    echo "Added '$new_key: $new_value' to '$parent_key'."
  else
    echo "Error: Failed to update '$config_file' with path '$full_path'."
    return 1
  fi

  echo "Configuration file '$config_file' updated successfully."
}

delete_yaml_field() {
  if [ "$#" -ne 2 ]; then
    echo "Usage: delete_yaml_field <config_file> <yaml_path>"
    return 1
  fi

  local config_file="$1"
  local yaml_path="$2"

  if [ ! -f "$config_file" ]; then
    echo "Error: File '$config_file' not found."
    return 1
  fi

  # Use yq to delete the field. The jq filter del() removes the key.
  yq -y "del(${yaml_path})" "$config_file" > tmp.yaml && mv tmp.yaml "$config_file"

  if [ $? -eq 0 ]; then
    echo "Deleted field ${yaml_path} from ${config_file}"
  else
    echo "Failed to update ${config_file}"
    return 1
  fi
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

#=========================== GANS training and preparation ==============================
train_gans(){
   local step_name="train_gans"
   export FAIRSEQ_ROOT=$FAIRSEQ_ROOT
   export KALDI_ROOT="$DIR_PATH/pykaldi/tools/kaldi"
   export KENLM_ROOT="$KENLM_ROOT"
   export PYTHONPATH="/$DIR_PATH:$PYTHONPATH"


   if is_completed "$step_name"; then
        log "Skipping gans training  (already completed)"
        return 0
    fi

    log "gans training."
    mark_in_progress "$step_name"
   

update_yaml_config "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/gan/w2vu.yaml" task.data="$CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled" task.text_data="$TEXT_OUTPUT/phones/" task.kenlm_path="$TEXT_OUTPUT/phones/lm.phones.filtered.04.bin" common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" model.code_penalty=2,4 model.gradient_penalty=1.5 model.smoothness_weight=0.5
add_to_existing_yaml "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/gan/w2vu.yaml" optimizer.groups.discriminator.optimizer lr [0.004]
add_to_existing_yaml "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/gan/w2vu.yaml" optimizer.groups.generator.optimizer lr [0.004]
delete_yaml_field "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/gan/w2vu.yaml" .optimizer.groups.generator.optimizer.amsgrad 
delete_yaml_field "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/gan/w2vu.yaml" .optimizer.groups.discriminator.optimizer.amsgrad

   PYTHONPATH=$FAIRSEQ_ROOT PREFIX=w2v_unsup_gan_xp fairseq-hydra-train \
    -m --config-dir "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/gan" \
    --config-name w2vu \
    task.data="$CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled" \
    task.text_data="$TEXT_OUTPUT/phones/" \
    task.kenlm_path="$TEXT_OUTPUT/phones/lm.phones.filtered.04.bin" \
    common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" \
    model.code_penalty=2,4 model.gradient_penalty=1.5,2.0 \
    model.smoothness_weight=0.5,0.75,1.0 'common.seed=range(0,5)' \
    2>&1 | tee $RESULTS_DIR/training1.log

   if [ $? -eq 0 ]; then
        mark_completed "$step_name"
        log "gans trained successfully"
    else
        log "ERROR: gans training failed"
        exit 1
    fi
}
