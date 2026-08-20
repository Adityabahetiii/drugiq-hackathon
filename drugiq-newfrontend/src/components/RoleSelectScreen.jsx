import { useState } from 'react'
import { FlaskConical, Lock, ShieldAlert, Stethoscope, User, ArrowRight } from './icons'

export default function RoleSelectScreen({ onSelectRole }) {
  const [doctorPass, setDoctorPass] = useState('')
  const [errorMsg, setErrorMsg] = useState('')
  const [selectedTab, setSelectedTab] = useState('patient') // 'patient' | 'doctor'

  const handleDoctorSubmit = (e) => {
    e?.preventDefault()
    if (doctorPass.trim() === 'doctor123') {
      setErrorMsg('')
      onSelectRole('doctor')
    } else {
      setErrorMsg('Incorrect credentials (use doctor123)')
    }
  }

  const handlePatientSubmit = () => {
    setErrorMsg('')
    onSelectRole('patient')
  }

  return (
    <div className="role-auth-overlay">
      <div className="role-auth-card">
        <div className="role-auth-header">
          <div className="role-auth-logo">
            <FlaskConical size={22} strokeWidth={2.2} />
          </div>
          <div className="role-auth-eyebrow">CLINICAL AI PORTAL</div>
          <h1 className="role-auth-title">Welcome to DrugIQ</h1>
          <p className="role-auth-sub">
            Select your access profile.
          </p>
        </div>

        <div className="role-cards-grid">
          {/* Doctor Card (Clinical Portal on Left) */}
          <div
            className={`role-option-card ${selectedTab === 'doctor' ? 'active' : ''}`}
            onClick={() => { setSelectedTab('doctor'); setErrorMsg('') }}
          >
            <div className="role-card-top">
              <div className="role-card-icon doctor-icon">
                <Stethoscope size={20} />
              </div>
              <span className="role-card-badge doctor-badge">Healthcare Professional</span>
            </div>
            <h2 className="role-card-name">Clinical Portal</h2>
            <p className="role-card-desc">
              Full prescribing information, clinical pharmacology, knowledge graphs, and multi-drug interaction analysis.
            </p>

            {selectedTab === 'doctor' && (
              <form
                className="doctor-auth-form"
                onSubmit={handleDoctorSubmit}
                onClick={(e) => e.stopPropagation()}
              >
                <div className="doctor-input-wrap">
                  <span className="input-lock-icon"><Lock size={15} /></span>
                  <input
                    type="password"
                    value={doctorPass}
                    onChange={(e) => { setDoctorPass(e.target.value); setErrorMsg('') }}
                    autoFocus
                  />
                </div>
                {errorMsg && (
                  <div className="role-auth-error">
                    <ShieldAlert size={14} /> {errorMsg}
                  </div>
                )}
                <button type="submit" className="role-enter-btn primary-btn">
                  Verify & Enter <ArrowRight size={15} />
                </button>
              </form>
            )}
          </div>

          {/* Patient Card (Patient Portal on Right) */}
          <div
            className={`role-option-card ${selectedTab === 'patient' ? 'active' : ''}`}
            onClick={() => { setSelectedTab('patient'); setErrorMsg('') }}
          >
            <div className="role-card-top">
              <div className="role-card-icon patient-icon">
                <User size={20} />
              </div>
              <span className="role-card-badge patient-badge">Patient Safe-Mode</span>
            </div>
            <h2 className="role-card-name">Patient Portal</h2>
            <p className="role-card-desc">
              Safe guidance on over-the-counter (OTC) medications with simplified dosing and safety guardrails.
            </p>
            {selectedTab === 'patient' && (
              <button
                type="button"
                className="role-enter-btn primary-btn"
                onClick={(e) => { e.stopPropagation(); handlePatientSubmit() }}
              >
                Continue as Patient <ArrowRight size={15} />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
