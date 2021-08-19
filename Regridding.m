%% Regrid cubes
% Load NetCDF file
cubes=ncread('MHWevent_cube.nc','Events');
% Fix dimentions (lon,lat,time)
cubes=permute(cubes, [2 3 1]); 
cubes=cubes(721:1060,393:640,:); % Select northeast Pacific
cubes=double(cubes); % Change data type

% Load and fix lon/lat arrays in (lon,lat,time) format
lo=ncread('NOAA_OI_climate_1983-2012.nc','lon'); % Load
lo=lo(721:1060); % Cut NE Pacific
la=ncread('NOAA_OI_climate_1983-2012.nc','lat');
la=la(393:640);

lo=repmat(lo,[1 248]);
la=repmat(la,[1 340])';

%%
cubes1=cubes(:,:,1);
% Make empty array
int_cubes=NaN*zeros(size(cubes1,1),size(cubes1,2),size(cubes1,3));

a=1;b=1; % Counters to use in the loop
step=10; % Step is number of pixels to average over for regridding
lon_start=1.8762500e+02; % Starting value for lon
lat_start=12.6250000; % Starting value for lat
idx_lon=find(lo(:,1)==lon_start); % Boolean search for start point in lon array
idx_lat=find(la(1,:)==lat_start); % Boolean search for start point in lat array

% 'For' loop to go through time slices
for kk=1:size(cubes1,3)
   % 'For' loop to go through lon
    for ii=idx_lon:step:(size(cubes1,1)-5)
        % 'For' loop to go through lat
        for jj=idx_lat:step:(size(cubes1,2)-5)
           % Make boolean indexes based on lon and lat cells
           idx1=find(lo(:,1)>=lo(ii,1)-1.25 & lo(:,1)<=lo(ii,1)+1.25);  
           idx2=find(la(1,:)>=la(1,jj)-1.25 & la(1,:)<=la(1,jj)+1.25);
           % Use idx1 & idx2 to position each pixel on the initial map
           % (cubes) for each time slice
           an1=squeeze(nanmean(nanmean(cubes1(idx1,idx2,kk))));
           % Write value into new array
           int_cubes(a,b,kk)=an1;
           clearvars an1
           % +1 steps for counters to keep up with the positioning in new array
           b=b+1; 
           
        end
        b=1;
        a=a+1; 
    
    end
    a=1;
  
end


clearvars -except int_sst time lo la int_cubes cubes

% Delete excessive NaNs
