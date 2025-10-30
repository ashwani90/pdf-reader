## Need to add the cron job so that the file read completes


### Cleanup json script

-  that runs and removes all Not Mentioned key/value pair
- 

## To process a pdf file

- Split the pdf using
-- url http://0.0.0.0:8000/split-pdf/?file_path=files/VBL-2023.pdf

- Scan pdf using url
-- http://0.0.0.0:8000/scan-pdfs/?directory_path=split_pdfs/files/VBL-2023/&orig_text_file=VBL-2023

- Save text file along with embedding in the db

- Create prompt file with all the questions

- Get answers to questions and create pdf report

