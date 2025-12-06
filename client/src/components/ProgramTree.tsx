import React from 'react';
import './ProgramTree.css';

interface Node {
  type: 'function' | 'terminal';
  name: string;
  children: Node[];
}

interface ProgramTreeProps {
  program: Node | null;
}

const ProgramTree: React.FC<ProgramTreeProps> = ({ program }) => {
  if (!program) {
    return null;
  }

  const renderNode = (node: Node) => {
    return (
      <div className={`tree-node ${node.type}`}>
        <div className="node-name">{node.name}</div>
        {node.children.length > 0 && (
          <div className="node-children">
            {node.children.map((child, index) => (
              <div key={index} className="child-container">
                {renderNode(child)}
              </div>
            ))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="program-tree">
      <h4>Best Program Tree</h4>
      {renderNode(program)}
    </div>
  );
};

export default ProgramTree;
