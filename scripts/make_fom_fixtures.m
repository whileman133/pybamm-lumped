function make_fom_fixtures(toolbox)
%MAKE_FOM_FIXTURES Export the toolbox cells the lumped.fom tests use.
%
%  Writes tests/data/<cell>.json with the toolbox's exportLumpedModel, for
%  the two public fixture cells.
%
%  -- Usage --
%    make_fom_fixtures('~/proj/BMS3-toolbox')     % from pybamm-lumped/scripts
%
%  -- Inputs --
%    toolbox - [char] BMS3-toolbox checkout.
%
%  See also EXPORTLUMPEDMODEL.

toolbox = char(java.io.File(strrep(toolbox, '~', getenv('HOME'))).getCanonicalPath());
addpath(fullfile(toolbox, 'UTILITY'), '-begin');
data = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'tests', 'data');
for c = {'cellNMC30', 'cellNMCMSMR'}
  exportLumpedModel(fullfile(toolbox, 'XLSX_CELLDEFS', [c{1} '.xlsx']), fullfile(data, [c{1} '.json']));
  fprintf('Wrote %s.json\n', c{1});
end
end
