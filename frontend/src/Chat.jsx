import React, { useState, useEffect, useRef } from 'react';
import './Chat.css';

import Markdown from 'react-markdown'

const StreamLangchain = ( {userID, handleDBUpdate} ) => {
    // State to store the input from the user
    const [input, setInput] = useState('');
    // State to store the responses/messages
    const [responses, setResponses] = useState([]);
    // Ref to manage the WebSocket connection
    const ws = useRef(null);
    // Ref to scroll to the latest message
    const messagesEndRef = useRef(null);
    // Maximum number of attempts to reconnect
    const [reconnectAttempts, setReconnectAttempts] = useState(0);
    const maxReconnectAttempts = 5;

    const [formheight, setFormheight] = useState('50px');

    // Function to setup the WebSocket connection and define event handlers
    const setupWebSocket = () => {
        ws.current = new WebSocket(`/ws/chat/${userID}`);
        let ongoingStream = null; // To track the ongoing stream's ID

        ws.current.onopen = () => {
            console.log("WebSocket connected!");
            setReconnectAttempts(0); // Reset reconnect attempts on successful connection
        };

        ws.current.onmessage = (event) => {
            const data = JSON.parse(event.data);
            let sender = "Geo Agent";
            console.log(data)
            
            //Handle different types of events from the WebSocket
            if (data.event === 'on_chat_model_start') {
                handleDBUpdate(true); // trigger side bar DB, 

                // When a new stream starts
                ongoingStream = { id: data.run_id, content: '' };
                setResponses(prevResponses => [...prevResponses, { sender, message: '', id: data.run_id }]);
            } else if (data.event === 'on_chat_model_stream' && ongoingStream && data.run_id === ongoingStream.id) {
                // During a stream, appending new chunks of data
                setResponses(prevResponses => prevResponses.map(msg =>
                  msg.id === data.run_id ? { ...msg, message: msg.message + data.data } : msg));
                    // msg.id === data.run_id ? { ...msg, message: msg.message + data.data.chunk } : msg));
            } else if (data.event === 'on_chat_model_end') {
                handleDBUpdate(false); // trigger side bar DB
            }

        };

        ws.current.onerror = (event) => {
            console.error("WebSocket error observed:", event);
        };

        ws.current.onclose = (event) => {
            console.log(`WebSocket is closed now. Code: ${event.code}, Reason: ${event.reason}`);
            handleReconnect();
        };
    };

    // Function to handle reconnection attempts with exponential backoff
    const handleReconnect = () => {
        if (reconnectAttempts < maxReconnectAttempts) {
            let timeout = Math.pow(2, reconnectAttempts) * 1000; // Exponential backoff
            setTimeout(() => {
                setupWebSocket(); // Attempt to reconnect
            }, timeout);
        } else {
            console.log("Max reconnect attempts reached, not attempting further reconnects.");
        }
    };

    // Effect hook to setup and cleanup the WebSocket connection
    useEffect(() => {
        setupWebSocket(); // Setup WebSocket on component mount

        return () => {
            if (ws.current.readyState === WebSocket.OPEN) {
                ws.current.close(); // Close WebSocket on component unmount
            }
        };
    }, []);

    // Effect hook to auto-scroll to the latest message
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [responses]);

    // Function to render each message
    const renderMessage = (response, index) => (
        <div key={index} className={`message ${response.sender}`}>
            <strong className="sender">{response.sender}</strong> 
            <div id='message-background'>
                {/* <p dangerouslySetInnerHTML={{ __html: response.message.replace(/\n/g, '<br />') }}></p> */}
                <Markdown>{response.message}</Markdown>
            </div>
        </div>
    );


    // Handler for input changes
    const handleInputChange = (e) => {
        const textarea = e.target;
        const container = textarea.closest('.input-form');
        
        textarea.style.height = 'auto';
        const newHeight = Math.min(textarea.scrollHeight, 300);
        textarea.style.height = `${newHeight}px`;
        container.scrollTop = container.scrollHeight;

        setFormheight(`${newHeight}px`)
        
        setInput(e.target.value);
    };

    const handleKeyDown = (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            handleSubmit(event);
        } else {
            setFormheight()
        }
    }

    // Handler for form submission
    const handleSubmit = (e) => {
        e.preventDefault();
        const userMessage = { sender: "You", message: input };
        setResponses(prevResponses => [...prevResponses, userMessage]);
        ws.current.send(JSON.stringify({ message: input })); // Send message through WebSocket
        setInput(''); // Clear input field
    };


    const fileInputRef = useRef(null);

    // Trigger the file input click
    const handleUploadButtonClick = () => {
        fileInputRef.current.click();
    };

    // Handle file selection and upload
    const handleUploadFileChange = async (e) => {
        const file = e.target.files[0];
        if (file) {
            console.log('File selected:', file);
            // Perform the file upload (e.g., using fetch or axios)
            // const formData = new FormData();
            // formData.append('file', file);
            // fetch('your-upload-endpoint', { method: 'POST', body: formData });
            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch(`/upload/${userID}`, {
                    method: 'POST',
                    body: formData,
                });

                if (response.ok) {
                    const data = await response.json();
                    console.log('File uploaded successfully:', data);

                    const userMessage = { sender: "You", message: `Uploaded file ${file.name}.` };
                    setResponses(prevResponses => [...prevResponses, userMessage]);
                    ws.current.send(JSON.stringify({ message: `User uploaded file ${file.name}.` }));
                } else {
                    console.error('File upload failed:', response.statusText);
                }
            } catch (error) {
                console.error('Error uploading file:', error);
            }
        }
    };


    return (
        <div className="chat-container">
            <div className="messages-container">
                {responses.map((response, index) => renderMessage(response, index))}
                <div ref={messagesEndRef} /> {/* Invisible element to help scroll into view */}
            </div>

            <div className='input-area'>
                <div className='upload-button'>
                    <button onClick={handleUploadButtonClick}>
                        Upload File
                    </button>
                    {/* Hidden file input */}
                    <input
                        type="file"
                        ref={fileInputRef}
                        style={{ display: 'none' }}
                        onChange={handleUploadFileChange}
                    />
                </div>
                <form className='input-form' onSubmit={handleSubmit}>
                    <textarea 
                        style={{ height: formheight }}
                        placeholder='Send a message' 
                        // onKeyUp={textarea_expand} 
                        onChange={handleInputChange}
                        onKeyDown={handleKeyDown}
                        value={input}
                        rows={1}>    
                    </textarea>
                    <button className='send-button' type="submit">Send</button>
                </form>
            </div>
            <div className='disclaimer'>Developed by GeoCloud group.</div>
            {/* <form onSubmit={handleSubmit} className="input-form" style={{ height: formheight }}>
                <div className="textarea-container">
                    <textarea
                        value={input}
                        onChange={handleInputChange}
                        onKeyDown={handleKeyDown}
                        placeholder="Type your message here..."
                        rows={1}
                    />
                </div>
                <button type="submit">Send</button>
            </form> */}
        </div>
    );
};

export default StreamLangchain;