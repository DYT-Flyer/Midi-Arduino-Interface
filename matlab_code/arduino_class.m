classdef arduino_class
    properties
        ard = arduino();
    end
    
    methods
        function turn_red_on(obj)
            writeDigitalPin(obj.ard,'D5',0);
            writeDigitalPin(obj.ard,'D6',0);
            writeDigitalPin(obj.ard,'D9',1);
            writeDigitalPin(obj.ard,'D10',0);
            writeDigitalPin(obj.ard,'D11',0);
        end
        
        function turn_blue_on(obj)
            writeDigitalPin(obj.ard,'D5',0);
            writeDigitalPin(obj.ard,'D6',0);
            writeDigitalPin(obj.ard,'D9',0);
            writeDigitalPin(obj.ard,'D10',1);
            writeDigitalPin(obj.ard,'D11',0);
        end
        
        function turn_yellow_on(obj)
            writeDigitalPin(obj.ard,'D5',0);
            writeDigitalPin(obj.ard,'D6',0);
            writeDigitalPin(obj.ard,'D9',0);
            writeDigitalPin(obj.ard,'D10',0);
            writeDigitalPin(obj.ard,'D11',1);
        end
        
        function turn_green_on(obj)
            writeDigitalPin(obj.ard,'D5',0);
            writeDigitalPin(obj.ard,'D6',1);
            writeDigitalPin(obj.ard,'D9',0);
            writeDigitalPin(obj.ard,'D10',0);
            writeDigitalPin(obj.ard,'D11',0);
        end
        
        function turn_white_on(obj)
            writeDigitalPin(obj.ard,'D5',1);
            writeDigitalPin(obj.ard,'D6',0);
            writeDigitalPin(obj.ard,'D9',0);
            writeDigitalPin(obj.ard,'D10',0);
            writeDigitalPin(obj.ard,'D11',0);
        end
    end
end