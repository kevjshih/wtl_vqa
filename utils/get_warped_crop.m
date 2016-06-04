function crop_resized = get_warped_crop(im, bbox, cropped_im_size)
% Extracts the cropped_im_size x cropped_im_size x 3 square warped image using
% coordinates from edge_box.padded_bbox
% output is single and normalized 
    
    DEBUG = 0;

    % load the image
    %    im = single(imread(sprintf('%s/%s/%s', im_dir, rec.folder, rec.filename)));
    
    % convert to 3 channels if it's grayscale
    if size(im, 3) == 1
        im = repmat(im, [1, 1, 3]);
    end
    [h, w, d] = size(im);
    
    %p_bbox= edge_box.padded_bbox;

    pad_startx = 0;
    if bbox(1) <= 0
        pad_startx = abs(bbox(1))+1; % +1 to account for the 0
    end
    pad_starty = 0;
    if bbox(2) <= 0
        pad_starty = abs(bbox(2))+1;
    end
    pad_endx = 0;
    if bbox(3) > w
        pad_endx = bbox(3) - w; % +1 to account for the 0
    end

    pad_endy = 0;
    if bbox(4) > h
        pad_endy = bbox(4) - h; % +1 to account for the 0
    end

    if pad_starty > 0 || pad_endy > 0 || pad_startx > 0 || pad_endx > 0
        new_h = h + pad_starty + pad_endy;
        new_w = w+pad_startx + pad_endx;
        im_padded = zeros(new_h, new_w, d);
        im_padded(pad_starty+1:pad_starty+h, pad_startx+1:pad_startx+w,:) = im;
        
        if DEBUG
            figure(2), imshow(im_padded)
        end
    
    else
        im_padded = im;
    end
    
    crop = im_padded(bbox(2)+pad_starty:bbox(4)+pad_starty, bbox(1)+pad_startx:bbox(3)+pad_startx,:);
    crop_resized = imresize(crop, [cropped_im_size, cropped_im_size], 'bilinear' ,'antialiasing', false);
    crop_resized = single(crop_resized);
    
end
