import json

from pathlib import Path

def save_chunk_records(chunk_records: list[dict], output_path: Path, mode="append"):
    modes = ["append", "overwrite"]
    if mode not in modes:
        raise ValueError ("Select a mode: append or overwrite")
    
    output_path = Path(output_path).resolve()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
        
    if mode=="append":
        write_mode = "a"
    elif mode=="overwrite":
        write_mode = "w"  
        
    with open(output_path, write_mode, encoding="utf-8") as f:
        for chunk_record in chunk_records:
            json_string=json.dumps(chunk_record)
            f.write(json_string + "\n")
            
def load_chunk_records(chunk_records_file: Path) -> tuple[list[dict], list[dict]]:
    chunk_records_file=Path(chunk_records_file).resolve()
    
    valid_records=[]
    load_errors=[]
    
    with open(chunk_records_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                valid_records.append(record)
                
            except json.JSONDecodeError as e:
                error_record = {
                    "line_number": line_num,
                    "raw_line": line,
                    "error_message": str(e)
                }
                load_errors.append(error_record)
                
    return valid_records, load_errors
                