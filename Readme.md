## Need to add the cron job so that the file read completes


### Cleanup json script

-  that runs and removes all Not Mentioned key/value pair

## To process a pdf file

### Split the pdf using
```
url http://0.0.0.0:8000/split-pdf/?file_path=files/VBL-2023.pdf
```

### Scan pdf using url
```
http://0.0.0.0:8000/scan-pdfs/?directory_path=split_pdfs/files/VBL-2023/&orig_text_file=VBL-2023
```
- Also change the name inside the file

### Save text file along with embedding in the db

- First insert the file using insert command
```
python rag-report_gen/insert_data_in_table.py
```
- Also update the name in the file first - TODO: Add dynamic param to get file name
- Split Larger chnuks into smaller ones using command:
```
python rag-report_gen/split_long_text.py
```


### Create prompt file with all the questions

#### Get answers to questions and create pdf report
- this will store the text embeddings in db
- COMPANY = "VBL-2023" needs to be changed to current file name
- Also change output file name
```
python rag-report_gen/rag_local_answering.py 
```


### Config
- the command ques embedding gen only runs once as it is only used to gen embeddings for stored questions



### To save prompts into db

```
python generate_reportable_text.py
```

### Get answers

```

python export_company_answers.py tata-motor
```