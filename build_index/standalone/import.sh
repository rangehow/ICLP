export MILVUS_URI="localhost:19530"

curl --request POST "http://${MILVUS_URI}/v2/vectordb/jobs/import/create" \
--header "Content-Type: application/json" \
--data-raw '{
    "files": [
        [
            "/mnt/rangehow/in-context-pretraining/build_embeds/8d3a3105-0ce0-447d-860f-093c1f20b020/1/$meta.npy"
        ],
        [
            "/mnt/rangehow/in-context-pretraining/build_embeds/8d3a3105-0ce0-447d-860f-093c1f20b020/1/vector.npy"
        ]
    ],
    "database":"ICLP",
    "collectionName": "ICLP"
}'