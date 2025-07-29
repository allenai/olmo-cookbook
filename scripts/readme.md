## all in one script spec

inputs:
- a file path to a file with lineseperated remote paths like `ai2-llm/pretraining-data/sources/finemath/finemath-3plus-decon-2/`
- a user specified remote prefix (defaults s3://) Ideally also support gs://
- a user specified input prefix, e.g. `ai2-llm/pretraining-data/sources/`
- a user specified output prefix, e.g. `ai2-llm/preprocessed/`
- a user specified local dir (defaults to /mnt/raid0)

commands :
- dowload: use s5cmd to download the input paths to the local dir, at the end report any errors in any of the downloads
- tokenize: tokenize locally from the input path to the output path using customizations to teh tokenization command specified in the tokenization gotchyas section bellow. At the end report any errors that occured for any of the paths
- upload: use s5cmd to upload the local tokenized data to the output prefix at the remote prefix,  report any errors in the uploads
- destination_paths: output uploaded paths to (one for each of the paths the user input) and provide wildcards (*.gz or **/*.gz) depending on if there are subdirs.

Design:
- keep the code as simple and concise as possible for easy review
- avoid silent failures (do not catch exceptions unless you can really handle them)



## tokenization gotchyas

problem: id type is not str
- if so check if the id field in first file is int and use 
        --fields.id_field_type int \

if the path ends with .gz or .jsonl or .json only use this as input to the tokenize command but drop the .gz or .jsonl.gz or .jsonl.gz, etc before using the path in the destination

Check if the path does not have *.gz immediately under it. If this is the case you will need to instead use all subdirs of the path as the seperate paths each with their own tokenization command.

problem: id field is not called "id" -> try backing off (using `--fields.id_field_name`) to any field with id in the name based on reading the first jsonl.gz line from the first file

Need enough disk space -> check that there is at least 10TB free disk space before attempting tokenization because the tokenization can fail silently if it runs out of space.

