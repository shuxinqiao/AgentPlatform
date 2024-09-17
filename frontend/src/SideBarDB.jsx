// RecordsList.js
import React, { useState, useEffect } from 'react';
import './SideBarDB.css'

import RecordBox from './DBRecord'

const RecordsList = ( {userID, updating} ) => {
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  

  useEffect(() => {

    fetch(`/db/${userID}`)
      .then(response => response.json())
      .then(data => {
        setRecords(data);
        setLoading(false);
      })
      .catch(error => {
        console.error('Error fetching data:', error);
        setLoading(false);
      });

  }, [userID, updating]);


  return (
    <div className='db-container'>
      <div>
        <h3>Records</h3>

        {records.map(record => (
          <RecordBox key={record.rowid} record={ record } />
        ))}

      </div>
      
    </div>
  );
};

export default RecordsList;