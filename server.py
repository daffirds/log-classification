from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse

from logs_classification import classify_csv

app = FastAPI()

@app.post("/classify/")
async def classify_logs(file: UploadFile):
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV.")
    
    try:
        # Perform classification
        output_file = classify_csv(file.file)
        
        print(f"File saved to {output_file}")
        return FileResponse(output_file, media_type='text/csv', filename="output.csv")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        file.file.close()