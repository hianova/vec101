use crate::core::Vec101Engine;
use std::env;
use std::fs;
use std::process::Command;

pub trait Vec101EngineReactExt {
    /// Generates a bash script via the engine and executes it.
    /// If the script fails, fetches `--help` for the failing command and tries again (ReAct loop).
    fn generate_and_execute(&mut self, prompt: &str) -> Result<(), String>;
}

impl Vec101EngineReactExt for Vec101Engine {
    fn generate_and_execute(&mut self, prompt: &str) -> Result<(), String> {
        println!(
            "\n[vec101 ReAct] Received Natural Language Task: \"{}\"",
            prompt
        );

        let mut current_prompt = format!(
            "Generate a pure bash script to accomplish this task:\nTask: {}\n\nOutput only the bash script inside a ```bash block.",
            prompt
        );

        let max_retries = 2;

        for attempt in 1..=max_retries {
            println!(
                "\n[vec101 ReAct] Attempt {}/{} - Transpiling to native script via vec101...",
                attempt, max_retries
            );

            // Generate script using batch_generate (with just one prompt)
            let responses = self.batch_generate(vec![current_prompt.clone()]);
            let script_content = match responses.first() {
                Some(content) if !content.is_empty() => content.clone(),
                _ => {
                    eprintln!("[vec101 ReAct Error] Failed to generate script from vec101.");
                    continue;
                }
            };

            println!("[vec101 ReAct] Received script from LLM.");

            // Write the script to a temporary executable file
            let mut temp_dir = env::temp_dir();
            temp_dir.push(format!("vec101_react_script_{}.sh", std::process::id()));

            fs::write(&temp_dir, &script_content)
                .map_err(|e| format!("Failed to write JIT script: {}", e))?;

            // Ensure the script is executable
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let mut perms = fs::metadata(&temp_dir)
                    .map_err(|e| e.to_string())?
                    .permissions();
                perms.set_mode(0o755);
                fs::set_permissions(&temp_dir, perms).map_err(|e| e.to_string())?;
            }

            println!(
                "[vec101 ReAct] Script compiled to {}. Executing...",
                temp_dir.display()
            );

            let output = Command::new(&temp_dir)
                .output()
                .map_err(|e| format!("Failed to execute JIT script: {}", e))?;

            if output.status.success() {
                println!("{}", String::from_utf8_lossy(&output.stdout));
                println!("[vec101 ReAct] Script execution SUCCESS.");
                return Ok(());
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                eprintln!("[vec101 ReAct Error] Execution failed: {}", stderr);

                // --- Self-Healing ReAct Loop ---
                // Extract the failing command
                let failing_cmd = if stderr.contains("command not found") {
                    let parts: Vec<&str> = stderr.split(':').collect();
                    if parts.len() >= 3 {
                        parts[parts.len() - 2]
                            .trim()
                            .split(' ')
                            .next()
                            .unwrap_or("")
                            .to_string()
                    } else {
                        stderr.split(' ').next().unwrap_or("unknown").to_string()
                    }
                } else {
                    "unknown".to_string()
                };

                println!(
                    "[vec101 ReAct] Attempting to learn usage of '{}' via --help...",
                    failing_cmd
                );

                let help_output = Command::new("bash")
                    .arg("-c")
                    .arg(format!("{} --help", failing_cmd))
                    .output();

                let help_text = match help_output {
                    Ok(out) => String::from_utf8_lossy(&out.stdout).to_string(),
                    Err(_) => "No manual entry available.".to_string(),
                };

                current_prompt = format!(
                    "Your previous bash script failed with error:\n{}\n\nHere is the manual for '{}':\n{}\n\nPlease fix the bash script.",
                    stderr, failing_cmd, help_text
                );
            }
        }

        Err("vec101 ReAct JIT compilation and execution failed after retries.".to_string())
    }
}
