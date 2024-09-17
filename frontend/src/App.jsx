import React, { useState, useEffect, useRef } from 'react';
import './App.css';

import SideBarDB from './SideBarDB'
import StreamLangchain from './Chat'


const Main = () => {
  const [userID, setUserID] = useState('user_' + new Date().toISOString().split('.')[0].replaceAll(":","-").replaceAll('-','_'))
  const [updating, setUpdating] = useState(false)

  const [isExpanded, setIsExpanded] = useState(false);
  const chatRef = useRef(null);

  const handleClick = () => {
    setIsExpanded(prevState => !prevState);
  };

  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.style.width = isExpanded ? 'calc(100% - 300px)' : '100%';
    }
  }, [isExpanded]);

  return (
    <div>
      <div className='main'>

        <div className='title'>
          <p id='logo'>Robo</p>
          <p id='chattitle'>{ userID }</p>
          <button onClick={handleClick}>Info</button>
        </div>

        <div className='chat-window'>
          <div className='chat' ref={chatRef}>
            <StreamLangchain userID={userID} handleDBUpdate={setUpdating}/>
          </div>
          <div className={`side-bar ${isExpanded ? 'expanded' : 'closed'}`}>
            <SideBarDB userID={userID} updating={updating}/> 
          </div>
        </div>
        
      </div>
    </div>
  );
};

export default Main; 