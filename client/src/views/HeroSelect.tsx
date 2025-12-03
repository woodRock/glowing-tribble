import React, { useEffect, useState } from 'react';
import type { StaffMember, Role, Shift } from '../types.ts';
import TeamSlot from '../components/TeamSlot';
import AvailableStaff from '../components/AvailableStaff';
import './HeroSelect.css';

interface SelectedTeam {
  bar: (StaffMember | null)[];
  restaurant: (StaffMember | null)[];
  "maitre'd": (StaffMember | null)[];
  "duty manager": (StaffMember | null)[];
}

const HeroSelect: React.FC = () => {
  const [allStaff, setAllStaff] = useState<StaffMember[]>([]);
  const [selectedTeam, setSelectedTeam] = useState<SelectedTeam>({
    bar: [null, null],
    restaurant: [null, null],
    "maitre'd": [null],
    "duty manager": [null],
  });
  const [generatedRoster, setGeneratedRoster] = useState<Shift[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch('http://localhost:4000/api/staff')
      .then((res) => res.json())
      .then((data) => {
        setAllStaff(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  const handleSelectStaff = (staffMember: StaffMember) => {
    if (selectedStaffIds.includes(staffMember.id)) return;
    const role = staffMember.roles[0];
    if (!role) return;
    const teamForRole = selectedTeam[role as keyof SelectedTeam];
    const emptySlotIndex = teamForRole.indexOf(null);
    if (emptySlotIndex !== -1) {
      const newTeamForRole = [...teamForRole];
      newTeamForRole[emptySlotIndex] = staffMember;
      setSelectedTeam({ ...selectedTeam, [role]: newTeamForRole });
    }
  };

  const handleDeselectStaff = (role: Role, index: number) => {
    const newTeamForRole = [...selectedTeam[role]];
    newTeamForRole[index] = null;
    setSelectedTeam({ ...selectedTeam, [role]: newTeamForRole });
  };

  const handleGenerateRoster = () => {
    fetch('http://localhost:4000/api/roster/generate', { method: 'POST' })
      .then(res => res.json())
      .then((generatedRoster: Shift[]) => {
        const newTeam: SelectedTeam = {
          bar: [null, null],
          restaurant: [null, null],
          "maitre'd": [null],
          "duty manager": [null],
        };

        generatedRoster.forEach(shift => {
          if (shift.staffMemberId) {
            const staffMember = allStaff.find(s => s.id === shift.staffMemberId);
            if (staffMember) {
              const teamForRole = newTeam[shift.role as keyof SelectedTeam];
              if(teamForRole) {
                const emptySlotIndex = teamForRole.indexOf(null);
                if (emptySlotIndex !== -1) {
                  teamForRole[emptySlotIndex] = staffMember;
                }
              }
            }
          }
        });

        setSelectedTeam(newTeam);
        setGeneratedRoster(generatedRoster);
      })
      .catch(err => console.error('Error generating roster:', err));
  };

  const selectedStaffIds = Object.values(selectedTeam).flat().filter(Boolean).map(s => s!.id);

  if (loading) return <div className="hero-select__message">Loading...</div>;
  if (error) return <div className="hero-select__message hero-select__message--error">Error: {error}</div>;

  return (
    <div className="hero-select">
      <div className="hero-select__team-selection">
        <div className="hero-select__title-container">
          <h2 className="hero-select__team-title">Your Team</h2>
          <button className="hero-select__generate-button" onClick={handleGenerateRoster}>
            Generate Roster
          </button>
        </div>
        <div className="hero-select__team-grid">
          {selectedTeam.bar.map((staff, i) => <TeamSlot key={`bar-${i}`} staffMember={staff} role="bar" onClick={() => handleDeselectStaff('bar', i)} />)}
          {selectedTeam.restaurant.map((staff, i) => <TeamSlot key={`restaurant-${i}`} staffMember={staff} role="restaurant" onClick={() => handleDeselectStaff('restaurant', i)} />)}
          {selectedTeam["maitre'd"].map((staff, i) => <TeamSlot key={`maitred-${i}`} staffMember={staff} role="maitre'd" onClick={() => handleDeselectStaff("maitre'd", i)} />)}
          {selectedTeam["duty manager"].map((staff, i) => <TeamSlot key={`dutymanager-${i}`} staffMember={staff} role="duty manager" onClick={() => handleDeselectStaff("duty manager", i)} />)}
        </div>
      </div>
      <AvailableStaff
        allStaff={allStaff}
        selectedStaffIds={selectedStaffIds}
        onSelectStaff={handleSelectStaff}
      />
    </div>
  );
};

export default HeroSelect;
