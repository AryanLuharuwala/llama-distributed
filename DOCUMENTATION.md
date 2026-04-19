Introduction

llama.cpp started as a single node server or cli to run inference on local servers, it operates on files that are of the type gguf, these can be quantized models or full weight models, models are trained or finetuned using unsloth
Unsloth TPP is a way to train lora adapters or the full weight training runs, 
The issue with rollouts is that they are generally distributed over a network switch rather than over the internet,
The network switch acts as a fast trasnfer for data directly using nccl or mpi transfer of the bytes, However there seems to be a lack of complete network transfer of packets wrapped in secure channels.
llama-distributed solves that using tcp transfer of the data in websocket relays peer to peer, 
there is a coordinator that handles signaling, relays, and scheduling of the nodes on the basis of the vram avaialbility and also token usage(TODO) these are to ensure maximum TDP usage by each individual GPU
Some GPUS may be vulkan some metal, so the protocol needs to be platform independent on windows, mac and linux, handled by a single web interface to manage the gpu pool,
the llama_decode public api uses the graph generated during model registration and uses model graph splitting to ensure only a few layers are run on a single node, this required the formation of 
a seperate engine, and also needs a seperate kernel for each individual architecture, as seen in the /models directory, using the apis we need to create a new file for all of them, the first node gets the
tokenizer and the token_embed while the last node will get the output_emb, a model splitting method is used to split a given gguf file into multiple small chunks that are sent to the nodes running those layers
the layers pass through and only pass between each other the tensors that the next layer will process, the actual decoding happens at the last layer itself, for places where the devices are behind a nat firewall
we will need turn servers and stun servers, these need to be implemented, a simple stun, turn can be implemented but for production scale we might have to make a better version of it,
the Ui can be like a chat room where we have a pool that shows up, on how many people are hosting what layers, we can later just have an idle node pass a token through a single layer, or even have tensor parallelizm that does all reduce and all gather
the Ui could have a easy to use dashboard with less than 4 clicks that gets the user up and running, it automatically sets up all the system on their local computer, making it very easy to join
a sharable link for their friends to join in, prompt caching, and last pass caching to ensure that pending requests are completed.


Objectives 
- llama_decode runs as a single layer instead of it needing to be hardcoded which layers are to be run through, configurable graph
- Coordinator schedules the layers on the nodes such that the utilization of each computer is equated and uses idle computers when more than a single node hosts the same layers
- Create a DNS like vercel so each pool can have its own OPEN_AI_BASE_URL, and allow the token authentication for ai requests, put it behind an email, or could have emailid.surds
- Turn server and Stun server implementation that passes the frames between the nodes, allows p2p connections between nat servers.
- Intuitive Web UI that users can connect their own server in less than 4 clicks
- a single download script like curl install.sh | bash or a .appimage for linux .dmg for apple or .msi installer for windows, autodetects the OS, installs the program, and the website generates a deeplink url that opens in the application
- The deep link connects the backend to the server and the frontend is disentangled from using a localhost address so it fetches the live data from our server.
