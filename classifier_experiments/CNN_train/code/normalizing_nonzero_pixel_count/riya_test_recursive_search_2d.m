x = imread('30672.bmp');
segNew = logical(x > 128); % thresholding

tempGraph = segNew;

skelIm = bwmorph(logical(tempGraph),'skel',Inf);
bridgedIm = bwmorph(skelIm,'bridge');
skelImFinal = bwmorph(logical(bridgedIm),'skel',Inf);
CC = bwconncomp(skelImFinal);
numPixels = cellfun(@numel,CC.PixelIdxList);
graphNodes = [];
graphPaths = {};

for num_pixels_idx = 1:length(numPixels)
    if numPixels(num_pixels_idx) > 1
        skelIm = bwmorph(logical(tempGraph),'skel',Inf);
        bridgedIm = bwmorph(skelIm,'bridge'); % bridge unconnected pixels
        skelImFinal = bwmorph(logical(bridgedIm),'skel',Inf);
        for temp_idx = 1:length(numPixels)
            if temp_idx ~= num_pixels_idx
                skelImFinal(CC.PixelIdxList{temp_idx}) = 0;
            end
        end
        
        %find end and branch points
        shape_endpts = bwmorph(skelImFinal,'endpoints');
        shape_branchdpts = bwmorph(skelImFinal,'branchpoints');
        ind_endpts = find(shape_endpts == 1);
        ind_branchpts = find(shape_branchdpts == 1);
        [I,J] = ind2sub(size(skelImFinal),ind_endpts);
        index_endpts = [I, J];
        [I,J] = ind2sub(size(skelImFinal),ind_branchpts);
        index_branchpts = [I, J];
        
        currEndPoint = index_endpts(2, :);
        currPoint = index_endpts(2, :);
        prevPoint = [];
        branchPointNodesAvoid = [];
        currLength = 0;
        graphNodes_temp = [];
        spacing = 1;
        currPath = [];
        graphPaths_temp = {};
        
        %[tempGraph, currEndPoint, currPoint, prevPoint, spacing, graphNodes_temp, currPath, graphPaths_temp] = riya_recursive_search_2d(tempGraph, currEndPoint, currPoint, prevPoint, branchPointNodesAvoid, currLength, spacing, graphNodes_temp, currPath, graphPaths_temp);
        [skelImFinal, currEndPoint, currPoint, prevPoint, spacing, graphNodes_temp, currPath, graphPaths_temp] = riya_recursive_search_2d(skelImFinal, currEndPoint, currPoint, prevPoint, branchPointNodesAvoid, currLength, spacing, graphNodes_temp, currPath, graphPaths_temp); 
        if isempty(graphNodes) == 1
            graphNodes = graphNodes_temp;
            graphPaths = graphPaths_temp;
        else
            graphNodes = [graphNodes; graphNodes_temp];
            graphPaths = [graphPaths, graphPaths_temp];
        end
    end
end

%make graph with weights
sourceNodes = size(graphNodes, 1);
sinkNodes = size(graphNodes, 1);
edgeLengths = size(graphNodes, 1);
for i = 1:size(graphNodes, 1)
    sourceNodes(i) = sub2ind(size(tempGraph), graphNodes(i,1), graphNodes(i,2));
    sinkNodes(i) = sub2ind(size(tempGraph), graphNodes(i,3), graphNodes(i,4));
    edgeLengths(i) = graphNodes(i,5);
end

hashTable = zeros(2*size(graphNodes, 1),2);
count = 1;
for i = 1:size(graphNodes, 1)
    id = sourceNodes(i);
    if i == 1
        hashTable(1,:) = [id, count];
        count = count + 1;
    else
        %check if key isn't already used
        if sum(hashTable(:,1) == id) == 0
            hashTable(count,:) = [id, count];
            count = count + 1;
        end
    end
end
for i = 1:size(graphNodes, 1)
    id = sinkNodes(i);
    %check if key isn't already used
    if sum(hashTable(:,1) == id) == 0
        hashTable(count,:) = [id, count];
        count = count + 1;
    end
end
[~, sourceNodesReduced] = max(sourceNodes == hashTable(:,1));
[~, sinkNodesReduced] = max(sinkNodes == hashTable(:,1));
names = cell(1, size(graphNodes, 1));
for i = 1:(count-1)
    names{i} = num2str(hashTable(i,1));
end

G = graph(sourceNodesReduced,sinkNodesReduced,edgeLengths,names);
plot(G,'EdgeLabel',G.Edges.Weight)
pairwiseDistance = distances(G);

%remove certain segments
skelImFinal = bwmorph(logical(bridgedIm),'skel',Inf);
skelImFinal_Deleted1 = skelImFinal;
skelImFinal_Deleted2 = skelImFinal;
skelImFinal_Deleted3 = skelImFinal;
path_to_delete1 = graphPaths{5};
path_to_delete2 = graphPaths{6};
path_to_delete3 = graphPaths{12};
for i = 1:size(path_to_delete1,1)
    skelImFinal_Deleted1(path_to_delete1(i, 1), path_to_delete1(i, 2)) = 0;
end
for i = 1:size(path_to_delete2,1)
    skelImFinal_Deleted2(path_to_delete2(i, 1), path_to_delete2(i, 2)) = 0;
end
for i = 1:size(path_to_delete3,1)
    skelImFinal_Deleted3(path_to_delete3(i, 1), path_to_delete3(i, 2)) = 0;
end
ha = tight_subplot(2,2,[.01 .01],[.01 .01],[.01 .01]);
axes(ha(1));
imshow(skelImFinal)
axes(ha(2));
imshow(skelImFinal_Deleted1)
axes(ha(3));
imshow(skelImFinal_Deleted2)
axes(ha(4));
imshow(skelImFinal_Deleted3)
