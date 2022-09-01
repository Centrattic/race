function [tempGraph, currEndPoint, currPoint, prevPoint, spacing, graphNodes, currPath, graphPaths] = riya_recursive_search_2d(tempGraph, currEndPoint, currPoint, prevPoint, branchPointNodesAvoid, currLength, spacing, graphNodes, currPath, graphPaths)

%possible movements
C = combnk([-1,0,1,-1,0,1],2);
C1 = unique(C, 'rows');
zeroInd = all(C1==0,2);
C1(zeroInd,:) = [];

%possible neighbors

d = size(C1);
d2 = size(currPoint);

disp(d);
disp('');

disp(currPoint);
disp(d2);

neighborIndex = C1 + currPoint;
zeroInd = any(neighborIndex<1,2) | any(neighborIndex(:,1)>size(tempGraph,1),2) | any(neighborIndex(:,2)>size(tempGraph,2),2);
neighborIndex(zeroInd,:) = [];
%remove previous point from possible
if size(prevPoint,1) > 0
    zeroInd = all(neighborIndex==prevPoint, 2);
    neighborIndex(zeroInd,:) = [];
end
%remove branch points from possible (if they exist)
for i = 1:size(branchPointNodesAvoid, 1)
    zeroInd = all(neighborIndex==branchPointNodesAvoid(i,:), 2);
    neighborIndex(zeroInd,:) = [];
end

linearInd = size(neighborIndex, 1);
for i = 1:size(neighborIndex, 1) 
    linearInd(i) = sub2ind(size(tempGraph), neighborIndex(i,1),neighborIndex(i,2));
end
indPath = find(tempGraph(linearInd) == 1);
tempGraph(currPoint(1), currPoint(2)) = 0;
currPath = [currPath; currPoint];

neighborMatrix = zeros(length(indPath), 2);
for i = 1:length(indPath)
    neighborMatrix(i, :) = neighborIndex(indPath(i), :);
end

%if only one way to go, move forward
if length(indPath) == 1
    if (neighborMatrix(:,1) == currPoint(1)) || (neighborMatrix(:,2) == currPoint(2))
        currLength = currLength + spacing;
    else
        currLength = currLength + sqrt(2*(spacing^2));
    end
    prevPoint = currPoint;
    currPoint = neighborIndex(indPath, :);
    branchPointNodesAvoid = [];
    [tempGraph, currEndPoint, currPoint, prevPoint, spacing, graphNodes, currPath, graphPaths] = riya_recursive_search_2d(tempGraph, currEndPoint, currPoint, prevPoint, branchPointNodesAvoid, currLength, spacing, graphNodes, currPath, graphPaths);
%if no more moves left, exist out of this recursion and move back down to
%previous branch point
elseif isempty(indPath)
    graphNodes = [graphNodes; currEndPoint(end, :), currPoint, currLength];
    graphPaths = [graphPaths, {currPath}];
    currPath = [];
%if two possible moves are next to each other, then not a branch point!
elseif (length(indPath) == 2) && ((all(abs(diff(neighborMatrix, 1, 1))==[1,0])) || (all(abs(diff(neighborMatrix, 1, 1))==[0, 1])))
    currLength = currLength + 1;
    prevPoint = currPoint;
    %find neighbor that is not on the diagonal to move to first
    if (sum(abs(neighborMatrix(1,:) - currPoint))) == 1
        currPoint = neighborMatrix(1,:);
    else
        currPoint = neighborMatrix(2,:);
    end
    branchPointNodesAvoid = [];
    [tempGraph, currEndPoint, currPoint, prevPoint, spacing, graphNodes, currPath, graphPaths] = riya_recursive_search_2d(tempGraph, currEndPoint, currPoint, prevPoint, branchPointNodesAvoid, currLength, spacing, graphNodes, currPath, graphPaths);
%if three possible moves are all in a line, then move up before branching!
elseif (length(indPath) == 3) && (all(neighborMatrix(:,1) == neighborMatrix(1,1)) || all(neighborMatrix(:,2) == neighborMatrix(1,2)))
    currLength = currLength + 1;
    prevPoint = currPoint;
    %find neighbor that is not on the diagonal to move to first
    if all(neighborMatrix(:,1) == neighborMatrix(1,1))
        tempMove = find(neighborMatrix(:,2) == currPoint(2));
        currPoint = neighborMatrix(tempMove,:);
    else
        tempMove = find(neighborMatrix(:,1) == currPoint(1));
        currPoint = neighborMatrix(tempMove,:);
    end
    branchPointNodesAvoid = [];
    [tempGraph, currEndPoint, currPoint, prevPoint, spacing, graphNodes, currPath, graphPaths] = riya_recursive_search_2d(tempGraph, currEndPoint, currPoint, prevPoint, branchPointNodesAvoid, currLength, spacing, graphNodes, currPath, graphPaths);
% at least two neighbors that are not touching indicates branch point
else
    graphNodes = [graphNodes; currEndPoint(end,:), currPoint, currLength];
    graphPaths = [graphPaths, {currPath}];
    currPath = [];
    currEndPoint = [currEndPoint; currPoint];
    prevPoint = currPoint;
    for i = 1:length(indPath)
        currLength = 0;
        if (neighborMatrix(i,1) == currPoint(1)) || (neighborMatrix(i,2) == currPoint(2))
            currLength = currLength + 1;
        else
            currLength = currLength + sqrt(2);
        end
        branchPointNodesAvoid = neighborIndex(indPath, :);
        currPoint = neighborMatrix(i,:);
        [tempGraph, currEndPoint, currPoint, prevPoint, spacing, graphNodes, currPath, graphPaths] = riya_recursive_search_2d(tempGraph, currEndPoint, currPoint, prevPoint, branchPointNodesAvoid, currLength, spacing, graphNodes, currPath, graphPaths);
    end
    currEndPoint = currEndPoint(1:(end-1), :);
end
