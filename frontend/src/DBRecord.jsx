import React, { useState, useEffect, useRef } from 'react';
import './DBRecord.css'

const RecordsItem = ( props ) => {
    const [showModal, setShowModal] = useState(false);
    const modalRef = useRef(null); // To reference the modal

    const record = props.record;

    const handleModal = () => {
        setShowModal(true)
    };

    const handleCloseModal = (e) => {
        if (modalRef.current && !modalRef.current.contains(e.target)) {
            setShowModal(false);
        }
    };

    useEffect(() => {
        // Add event listener when the modal is shown
        if (showModal) {
            document.addEventListener('mousedown', handleCloseModal);
        } else {
            document.removeEventListener('mousedown', handleCloseModal);
        }

        // Cleanup the event listener
        return () => {
            document.removeEventListener('mousedown', handleCloseModal);
        };
    }, [showModal]);


    return (
        <div>
            <button className="rounded-box" onClick={handleModal}>
                <div className="icon-box">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">
                        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm1-13h-2v6h2V7zm0 8h-2v2h2v-2z"/>
                    </svg>
                </div>
                <div className="separator"></div>
                <div className="text-box">
                    <p>{ 'ID: ' + record.rowid }</p>
                    <p>{ 'Type: ' + record.type }</p>
                    {/* <p>{ record.description }</p> */}
                </div>
            </button>
            <div className={`record-modal ${showModal ? 'open' : ''}`}>
                <div className='modal-content' ref={modalRef}>
                    <span className="modal-close-area" onClick={() => setShowModal(false)}>X</span>
                    <p>{ 'ID: ' + record.rowid }</p>
                    <p>{ 'Type: ' + record.type }</p>
                    <p>{ 'Description: \n' + record.description }</p>
                    
                </div>
            </div>
        </div>
    );
};

export default RecordsItem;